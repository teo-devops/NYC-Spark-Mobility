"""
src/etl/raw_to_silver.py
────────────────────────
Job Spark Batch: Bronze → Silver
· Carga Parquets raw con StructType estricto
· Normaliza columnas Yellow y Green a esquema común
· Join con taxi_zone_lookup para añadir Borough y Zone
· Aplica filtros limpiar_con_logica_nyc()
· Calcula duration_min
· Registra métricas de calidad en MLflow
· Escribe Silver particionado por fleet y month
"""

import os
import sys

sys.path.insert(0, "/opt/spark/src")

import mlflow
from pyspark.sql import functions as F

from common.schemas import YELLOW_SCHEMA, GREEN_SCHEMA, ZONE_SCHEMA
from common.utils import get_spark, apply_nyc_filters, log_etl_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
BRONZE_PATH  = "/opt/spark/data/bronze"
SILVER_PATH  = "/opt/spark/data/silver"
ZONES_PATH   = f"{BRONZE_PATH}/taxi_zone_lookup.csv"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT   = "nyc_taxi_etl"

# ── Datasets a procesar ────────────────────────────────────────────────────────
DATASETS = [
    ("yellow", "2025-03"),
    ("yellow", "2025-11"),
    ("green",  "2025-03"),
    ("green",  "2025-11"),
]


def load_zones(spark):
    return (
        spark.read
        .option("header", "true")
        .schema(ZONE_SCHEMA)
        .csv(ZONES_PATH)
    )


def normalize_yellow(df):
    """Renombra columnas Yellow al esquema Silver común."""
    return (
        df
        .withColumn("fleet", F.lit("yellow"))
        .withColumnRenamed("tpep_pickup_datetime",  "pickup_datetime")
        .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")
    )


def normalize_green(df):
    """Renombra columnas Green al esquema Silver común."""
    return (
        df
        .withColumn("fleet", F.lit("green"))
        .withColumnRenamed("lpep_pickup_datetime",  "pickup_datetime")
        .withColumnRenamed("lpep_dropoff_datetime", "dropoff_datetime")
    )


def enrich_with_zones(df, zones_df):
    """Join con zonas para PU y DO."""
    pu = zones_df.select(
        F.col("LocationID").alias("PULocationID"),
        F.col("Borough").alias("PUBorough"),
        F.col("Zone").alias("PUZone"),
    )
    do_ = zones_df.select(
        F.col("LocationID").alias("DOLocationID"),
        F.col("Borough").alias("DOBorough"),
        F.col("Zone").alias("DOZone"),
    )
    return df.join(pu, on="PULocationID", how="left").join(do_, on="DOLocationID", how="left")


def add_duration(df):
    return df.withColumn(
        "duration_min",
        (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
    )


def process_dataset(spark, fleet: str, month: str, zones_df):
    schema = YELLOW_SCHEMA if fleet == "yellow" else GREEN_SCHEMA
    path   = f"{BRONZE_PATH}/{fleet}_tripdata_{month}.parquet"

    raw = spark.read.schema(schema).parquet(path)
    raw_count = raw.count()

    # Normalizar → columnas comunes
    normalized = normalize_yellow(raw) if fleet == "yellow" else normalize_green(raw)

    # Añadir duration antes de filtrar (filtro usa duration_min)
    with_duration = add_duration(normalized)

    # Filtros NYC TLC
    clean = apply_nyc_filters(with_duration, fleet)
    clean_count = clean.count()

    # Enriquecer con zonas
    enriched = enrich_with_zones(clean, zones_df)

    # Seleccionar columnas Silver finales
    silver = enriched.select(
        "fleet", "VendorID",
        "pickup_datetime", "dropoff_datetime",
        "PULocationID", "DOLocationID",
        "PUBorough", "DOBorough", "PUZone", "DOZone",
        "trip_distance", "fare_amount", "total_amount",
        "passenger_count", "payment_type", "duration_min",
    ).withColumn("month", F.lit(month))

    return silver, raw_count, clean_count


def main():
    spark = get_spark("ETL_Bronze_to_Silver")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    zones_df = load_zones(spark)
    zones_df.cache()

    with mlflow.start_run(run_name="raw_to_silver_batch") as run:

        mlflow.log_param("datasets", str(DATASETS))
        mlflow.log_param("silver_path", SILVER_PATH)

        all_silver = []

        for fleet, month in DATASETS:
            silver, raw_count, clean_count = process_dataset(spark, fleet, month, zones_df)
            log_etl_metrics(run, raw_count, clean_count, fleet, month)
            all_silver.append(silver)

        # Union de los 4 datasets → un único Silver particionado
        silver_full = all_silver[0]
        for df in all_silver[1:]:
            silver_full = silver_full.union(df)

        (
            silver_full
            .repartition(8, "fleet", "month")
            .write
            .mode("overwrite")
            .partitionBy("fleet", "month")
            .parquet(SILVER_PATH)
        )

        total_rows = silver_full.count()
        mlflow.log_metric("silver_total_rows", total_rows)

        print(f"✅ Silver escrito en {SILVER_PATH}  ({total_rows:,} filas)")

    spark.stop()


if __name__ == "__main__":
    main()
