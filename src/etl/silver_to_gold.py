"""
src/etl/silver_to_gold.py
──────────────────────────
Job Spark Batch: Silver → Gold
· Feature engineering idéntico al Notebook 02_predictive.ipynb de taxi-app-ai
· Genera:
    gold/features/    → tabla ABT para entrenamiento del modelo
    gold/dist_media/  → distancia media histórica por par PU/DO (para Streamlit)
    gold/agg_hourly/  → KPIs agregados por hora (para EDA distribuido)
"""

import os
import sys

sys.path.insert(0, "/opt/spark/src")

import mlflow
from pyspark.sql import functions as F

from common.utils import get_spark

SILVER_PATH      = "/opt/spark/data/silver"
GOLD_FEATURES    = "/opt/spark/data/gold/features"
GOLD_DIST_MEDIA  = "/opt/spark/data/gold/dist_media"
GOLD_AGG_HOURLY  = "/opt/spark/data/gold/agg_hourly"
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT       = "nyc_taxi_etl"


def build_features(df):
    """
    Replica exacta del feature engineering de 02_predictive.ipynb,
    traducido a Spark SQL functions.
    Features: hour, day_of_week, month_num, is_weekend, fleet_int
    Target:   fare_amount, duration_min
    """
    return (
        df
        .withColumn("hour",        F.hour("pickup_datetime"))
        .withColumn("day_of_week", F.dayofweek("pickup_datetime"))   # 1=Dom, 7=Sáb
        .withColumn("month_num",   F.month("pickup_datetime"))
        .withColumn("is_weekend",  (F.dayofweek("pickup_datetime").isin(1, 7)).cast("int"))
        .withColumn("fleet_int",   (F.col("fleet") == "yellow").cast("int"))
        # Ratios extra (para EDA gold)
        .withColumn("usd_per_km",  F.col("fare_amount") / (F.col("trip_distance") * 1.60934))
        .withColumn("min_per_km",  F.col("duration_min") / (F.col("trip_distance") * 1.60934))
    )


def build_dist_media(df):
    """
    Tabla de distancia media histórica por par (PULocationID, DOLocationID).
    Usada por Streamlit cuando el usuario no introduce distancia manualmente.
    Equivale al dist_media.csv de taxi-app-ai, pero en Parquet distribuido.
    """
    return (
        df
        .groupBy("PULocationID", "DOLocationID")
        .agg(
            F.avg("trip_distance").alias("dist_media_miles"),
            F.avg("fare_amount").alias("fare_media"),
            F.avg("duration_min").alias("duration_media_min"),
            F.count("*").alias("n_trips"),
        )
        .filter(F.col("n_trips") >= 5)   # mínimo estadístico
    )


def build_agg_hourly(df):
    """Tabla de KPIs por hora del día y flota para dashboards."""
    return (
        df
        .groupBy("fleet", "hour")
        .agg(
            F.avg("fare_amount").alias("avg_fare"),
            F.avg("duration_min").alias("avg_duration_min"),
            F.avg("trip_distance").alias("avg_distance_miles"),
            F.avg("usd_per_km").alias("avg_usd_per_km"),
            F.count("*").alias("n_trips"),
        )
        .orderBy("fleet", "hour")
    )


def main():
    spark = get_spark("ETL_Silver_to_Gold")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    silver = spark.read.parquet(SILVER_PATH)

    with mlflow.start_run(run_name="silver_to_gold") as run:

        # ── Features ABT ───────────────────────────────────────────────────────
        features = build_features(silver)

        abt_cols = [
            "PULocationID", "DOLocationID", "trip_distance",
            "hour", "day_of_week", "month_num", "is_weekend",
            "fleet_int", "fare_amount", "duration_min",
        ]
        abt = features.select(*abt_cols).dropna()

        (
            abt
            .repartition(8)
            .write
            .mode("overwrite")
            .parquet(GOLD_FEATURES)
        )
        mlflow.log_metric("gold_features_rows", abt.count())
        print(f"✅ Gold features: {abt.count():,} filas → {GOLD_FEATURES}")

        # ── Dist media ─────────────────────────────────────────────────────────
        dist_media = build_dist_media(silver)
        (
            dist_media
            .write
            .mode("overwrite")
            .parquet(GOLD_DIST_MEDIA)
        )
        print(f"✅ Gold dist_media: {dist_media.count():,} pares → {GOLD_DIST_MEDIA}")

        # ── Agg hourly ─────────────────────────────────────────────────────────
        agg_hourly = build_agg_hourly(features)
        (
            agg_hourly
            .write
            .mode("overwrite")
            .parquet(GOLD_AGG_HOURLY)
        )
        print(f"✅ Gold agg_hourly → {GOLD_AGG_HOURLY}")

        mlflow.log_param("gold_features_path", GOLD_FEATURES)
        mlflow.log_param("gold_dist_path",     GOLD_DIST_MEDIA)

    spark.stop()


if __name__ == "__main__":
    main()
