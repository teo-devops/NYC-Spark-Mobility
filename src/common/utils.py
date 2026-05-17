"""
src/common/utils.py
───────────────────
Utilidades compartidas entre todos los jobs Spark.
· get_spark()  — SparkSession pre-configurada con Hive + MLflow
· haversine_udf — distancia geográfica en millas (igual que taxi-app-ai)
· log_etl_metrics — registra métricas de limpieza en MLflow
"""

import os
import math
import logging
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nyc_taxi")


# ── SparkSession factory ───────────────────────────────────────────────────────

def get_spark(
    app_name: str = "NYCTaxiLab",
    master: Optional[str] = None,
    extra_config: Optional[dict] = None,
) -> SparkSession:
    """
    Crea o recupera la SparkSession con:
    · soporte Hive (metastore externo)
    · MLflow tracking URI desde env
    · Event log habilitado para History Server
    """
    master = master or os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .enableHiveSupport()
        .config("spark.sql.warehouse.dir", "/opt/spark/warehouse")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/opt/spark/spark-events")
        .config("spark.mlflow.trackingUri", mlflow_uri)
        # Parquet — disable vectorized reader so INT32/INT64 mismatches are coerced silently
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if extra_config:
        for k, v in extra_config.items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession iniciada: %s — master: %s", app_name, master)
    return spark


# ── Haversine UDF ──────────────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en millas entre dos coordenadas geográficas."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 3958.8  # radio Tierra en millas
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


haversine_udf = F.udf(_haversine_miles, DoubleType())


# ── Limpieza NYC TLC (lógica de taxi-app-ai) ──────────────────────────────────

FILTER_RULES = {
    "trip_distance":  (0.18, 100.0),   # millas
    "fare_amount":    (3.0,  600.0),   # USD
    "duration_min":   (1.0,  300.0),   # minutos
}


def apply_nyc_filters(df: DataFrame, fleet: str) -> DataFrame:
    """
    Aplica los filtros de limpiar_con_logica_nyc() como columnas Spark.
    Equivalente exacto a la lógica del Notebook 01_dta.ipynb de taxi-app-ai.
    """
    df = (
        df
        .filter(F.col("trip_distance").between(*FILTER_RULES["trip_distance"]))
        .filter(F.col("fare_amount").between(*FILTER_RULES["fare_amount"]))
        .filter(F.col("duration_min").between(*FILTER_RULES["duration_min"]))
        .filter(F.col("PULocationID") <= 263)
        .filter(F.col("DOLocationID") <= 263)
        .filter(F.col("PULocationID") != F.col("DOLocationID"))
        .filter(F.col("pickup_datetime").isNotNull())
        .filter(F.col("dropoff_datetime").isNotNull())
        .filter(F.col("pickup_datetime") < F.col("dropoff_datetime"))
    )
    logger.info("Filtros NYC aplicados para flota: %s", fleet)
    return df


# ── ETL Metrics para MLflow ────────────────────────────────────────────────────

def log_etl_metrics(run, raw_count: int, clean_count: int, fleet: str, month: str) -> None:
    """
    Registra métricas de calidad del ETL en el run MLflow activo.
    En MLflow 2.x las métricas se loguean con mlflow.log_metric(), no en el objeto run.
    """
    import mlflow as _mlflow
    removed = raw_count - clean_count
    pct_removed = removed / raw_count * 100 if raw_count > 0 else 0.0

    _mlflow.log_metric(f"{fleet}_{month}_raw_count",   raw_count)
    _mlflow.log_metric(f"{fleet}_{month}_clean_count", clean_count)
    _mlflow.log_metric(f"{fleet}_{month}_removed_pct", round(pct_removed, 2))

    logger.info(
        "[%s %s] raw=%d  clean=%d  eliminados=%.1f%%",
        fleet, month, raw_count, clean_count, pct_removed,
    )
