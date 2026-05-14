"""
src/etl/streaming_job.py
─────────────────────────
Spark Structured Streaming: streaming_source/ → Gold (micro-batch)

Simula la llegada de datos en tiempo real moviendo Parquets a data/streaming_source/.
Cada micro-batch aplica los mismos filtros Silver y escribe en Gold/streaming/.

Uso:
    docker exec spark-master spark-submit \\
      --master spark://spark-master:7077 \\
      /opt/spark/src/etl/streaming_job.py

    # Simular llegada de datos:
    cp data/bronze/yellow_tripdata_2025-03.parquet data/streaming_source/batch_001.parquet
"""

import os
import sys

sys.path.insert(0, "/opt/spark/src")

from pyspark.sql import functions as F

from common.schemas import YELLOW_SCHEMA
from common.utils import get_spark, apply_nyc_filters

SOURCE_PATH    = "/opt/spark/data/streaming_source"
GOLD_STREAM    = "/opt/spark/data/gold/streaming"
CHECKPOINT_DIR = "/opt/spark/data/gold/_checkpoints/streaming"
TRIGGER_SECS   = 30   # micro-batch cada 30 segundos


def add_duration(df):
    return df.withColumn(
        "duration_min",
        (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
    )


def add_features(df):
    return (
        df
        .withColumn("hour",        F.hour("pickup_datetime"))
        .withColumn("day_of_week", F.dayofweek("pickup_datetime"))
        .withColumn("month_num",   F.month("pickup_datetime"))
        .withColumn("is_weekend",  (F.dayofweek("pickup_datetime").isin(1, 7)).cast("int"))
        .withColumn("fleet_int",   F.lit(1))   # streaming solo amarillo en este ejemplo
        .withColumn("ingested_at", F.current_timestamp())
    )


def main():
    spark = get_spark(
        "Streaming_HotFolder_to_Gold",
        extra_config={
            "spark.sql.streaming.schemaInference": "true",
        },
    )

    # ── Leer stream desde hot-folder ──────────────────────────────────────────
    raw_stream = (
        spark.readStream
        .schema(YELLOW_SCHEMA)                       # StructType obligatorio en streaming
        .option("maxFilesPerTrigger", 1)             # un archivo por micro-batch
        .parquet(SOURCE_PATH)
    )

    # ── Transformaciones idénticas al batch ──────────────────────────────────
    normalized = (
        raw_stream
        .withColumn("fleet",            F.lit("yellow"))
        .withColumnRenamed("tpep_pickup_datetime",  "pickup_datetime")
        .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")
    )

    with_duration = add_duration(normalized)
    clean         = apply_nyc_filters(with_duration, "yellow")
    enriched      = add_features(clean)

    output_cols = [
        "fleet", "PULocationID", "DOLocationID", "trip_distance",
        "fare_amount", "duration_min",
        "hour", "day_of_week", "month_num", "is_weekend", "fleet_int",
        "pickup_datetime", "ingested_at",
    ]
    output = enriched.select(*output_cols)

    # ── Escribir en Gold/streaming (append) ──────────────────────────────────
    query = (
        output.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path",          GOLD_STREAM)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .partitionBy("hour")
        .trigger(processingTime=f"{TRIGGER_SECS} seconds")
        .start()
    )

    print(f"🚀 Streaming job arrancado. Escuchando en: {SOURCE_PATH}")
    print(f"   Escribiendo en: {GOLD_STREAM}")
    print(f"   Checkpoint: {CHECKPOINT_DIR}")

    query.awaitTermination()


if __name__ == "__main__":
    main()
