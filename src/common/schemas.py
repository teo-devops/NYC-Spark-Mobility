"""
src/common/schemas.py
─────────────────────
Definición centralizada de StructType para los tres datasets NYC TLC.
Reutiliza la lógica de limpieza de taxi-app-ai con tipado explícito de Spark.
"""

from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, FloatType, DoubleType,
    StringType, TimestampType,
)

# ── Yellow Taxi ────────────────────────────────────────────────────────────────
# Actual types from yellow_tripdata_2024-01.parquet (verified via pyarrow):
#   VendorID=int32, passenger_count=int64, RatecodeID=int64,
#   PULocationID=int32, DOLocationID=int32, payment_type=int64,
#   trip_distance/fare_*/amounts = double
# NOTE: raw_to_silver.py reads WITHOUT explicit schema (Parquet is self-describing).
#       This StructType is kept as documentation only.
YELLOW_SCHEMA = StructType([
    StructField("VendorID",             IntegerType(),   True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime",TimestampType(), True),
    StructField("passenger_count",      LongType(),      True),
    StructField("trip_distance",        DoubleType(),    True),
    StructField("RatecodeID",           LongType(),      True),
    StructField("store_and_fwd_flag",   StringType(),    True),
    StructField("PULocationID",         IntegerType(),   True),
    StructField("DOLocationID",         IntegerType(),   True),
    StructField("payment_type",         LongType(),      True),
    StructField("fare_amount",          DoubleType(),    True),
    StructField("extra",                DoubleType(),    True),
    StructField("mta_tax",              DoubleType(),    True),
    StructField("tip_amount",           DoubleType(),    True),
    StructField("tolls_amount",         DoubleType(),    True),
    StructField("improvement_surcharge",DoubleType(),    True),
    StructField("total_amount",         DoubleType(),    True),
    StructField("congestion_surcharge", DoubleType(),    True),
    StructField("Airport_fee",          DoubleType(),    True),
])

# ── Green Taxi ─────────────────────────────────────────────────────────────────
GREEN_SCHEMA = StructType([
    StructField("VendorID",             IntegerType(),   True),
    StructField("lpep_pickup_datetime", TimestampType(), True),
    StructField("lpep_dropoff_datetime",TimestampType(), True),
    StructField("store_and_fwd_flag",   StringType(),    True),
    StructField("RatecodeID",           DoubleType(),    True),
    StructField("PULocationID",         IntegerType(),   True),
    StructField("DOLocationID",         IntegerType(),   True),
    StructField("passenger_count",      DoubleType(),    True),
    StructField("trip_distance",        DoubleType(),    True),
    StructField("fare_amount",          DoubleType(),    True),
    StructField("extra",                DoubleType(),    True),
    StructField("mta_tax",              DoubleType(),    True),
    StructField("tip_amount",           DoubleType(),    True),
    StructField("tolls_amount",         DoubleType(),    True),
    StructField("ehail_fee",            DoubleType(),    True),
    StructField("improvement_surcharge",DoubleType(),    True),
    StructField("total_amount",         DoubleType(),    True),
    StructField("payment_type",         DoubleType(),    True),
    StructField("trip_type",            DoubleType(),    True),
    StructField("congestion_surcharge", DoubleType(),    True),
])

# ── Taxi Zone Lookup ───────────────────────────────────────────────────────────
ZONE_SCHEMA = StructType([
    StructField("LocationID", IntegerType(), True),
    StructField("Borough",    StringType(),  True),
    StructField("Zone",       StringType(),  True),
    StructField("service_zone", StringType(), True),
])

# ── Silver (output schema compartido) ─────────────────────────────────────────
SILVER_SCHEMA = StructType([
    StructField("fleet",             StringType(),    False),  # "yellow" | "green"
    StructField("VendorID",          IntegerType(),   True),
    StructField("pickup_datetime",   TimestampType(), False),
    StructField("dropoff_datetime",  TimestampType(), False),
    StructField("PULocationID",      IntegerType(),   False),
    StructField("DOLocationID",      IntegerType(),   False),
    StructField("PUBorough",         StringType(),    True),
    StructField("DOBorough",         StringType(),    True),
    StructField("PUZone",            StringType(),    True),
    StructField("DOZone",            StringType(),    True),
    StructField("trip_distance",     DoubleType(),    False),
    StructField("fare_amount",       DoubleType(),    False),
    StructField("total_amount",      DoubleType(),    True),
    StructField("passenger_count",   DoubleType(),    True),
    StructField("payment_type",      IntegerType(),   True),
    StructField("duration_min",      DoubleType(),    False),  # calculado
])
