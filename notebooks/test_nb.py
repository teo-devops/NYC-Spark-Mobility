import os, subprocess, sys
sys.path.insert(0, "/opt/spark/src")

SPARK_MASTER   = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Paths inside Spark containers (used for spark-submit jobs)
BRONZE_PATH    = "/opt/spark/data/bronze"
SILVER_PATH    = "/opt/spark/data/silver"
GOLD_PATH      = "/opt/spark/data/gold"

# Paths inside this Jupyter container (used for local file checks)
WORK_DIR       = "/home/jovyan/work"
BRONZE_LOCAL   = f"{WORK_DIR}/data/bronze"
GOLD_LOCAL     = f"{WORK_DIR}/data/gold"

print(f"Spark master : {SPARK_MASTER}")
print(f"MLflow URI   : {MLFLOW_URI}")

from common.utils import get_spark

spark = get_spark("Pipeline_Notebook", master=SPARK_MASTER)
spark

import glob

parquets = glob.glob(f"{BRONZE_PATH}/*.parquet")
csvs     = glob.glob(f"{BRONZE_PATH}/*.csv")

print("Parquets encontrados:")
for f in sorted(parquets):
    size_mb = os.path.getsize(f) / 1_048_576
    print(f"  {os.path.basename(f):45s}  {size_mb:.1f} MB")

print("\nCSVs encontrados:")
for f in sorted(csvs):
    print(f"  {os.path.basename(f)}")

assert len(parquets) >= 1, "❌ No hay Parquets en data/bronze — descarga los datos NYC TLC primero"
assert any("taxi_zone_lookup" in f for f in csvs), "❌ Falta taxi_zone_lookup.csv en data/bronze"
print("\n✅ Bronze OK")

# Lanza raw_to_silver.py via spark-submit desde el contenedor Spark Master.
# Si ya tienes Silver generado puedes saltar esta celda.

result = subprocess.run(
    [
        "docker", "exec", "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master", SPARK_MASTER,
        "/opt/spark/src/etl/raw_to_silver.py",
    ],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("raw_to_silver.py falló")
print("✅ Bronze → Silver completado")

# Validación Silver
silver = spark.read.parquet(SILVER_PATH)
print(f"Filas Silver : {silver.count():,}")
print(f"Particiones  : {silver.rdd.getNumPartitions()}")
silver.printSchema()
silver.show(5, truncate=False)

# Distribución por flota y mes
from pyspark.sql import functions as F

silver.groupBy("fleet", "month").count().orderBy("fleet", "month").show()

result = subprocess.run(
    [
        "docker", "exec", "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master", SPARK_MASTER,
        "/opt/spark/src/etl/silver_to_gold.py",
    ],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("silver_to_gold.py falló")
print("✅ Silver → Gold completado")

# Validación Gold — features ABT
gold_features = spark.read.parquet(f"{GOLD_PATH}/features")
print(f"Gold features : {gold_features.count():,} filas")
gold_features.printSchema()
gold_features.describe("fare_amount", "duration_min", "trip_distance").show()

# Validación Gold — dist_media (usada por Streamlit)
dist_media = spark.read.parquet(f"{GOLD_PATH}/dist_media")
print(f"Pares PU/DO únicos : {dist_media.count():,}")
dist_media.orderBy(F.col("n_trips").desc()).show(10)

# Validación Gold — agg_hourly
import plotly.express as px
import pandas as pd

agg = spark.read.parquet(f"{GOLD_PATH}/agg_hourly").toPandas()
fig = px.line(
    agg, x="hour", y="avg_fare", color="fleet",
    title="Tarifa media por hora del día",
    labels={"avg_fare": "Tarifa media ($)", "hour": "Hora"}
)
fig.show()

result = subprocess.run(
    [
        "docker", "exec", "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master", SPARK_MASTER,
        "/opt/spark/src/models/train.py",
    ],
    capture_output=True, text=True
)
print(result.stdout[-4000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("train.py falló")
print("✅ Entrenamiento completado")

# Consulta métricas del último run en MLflow
import mlflow

mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

for exp_name in ["nyc_taxi_model"]:
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"Experimento '{exp_name}' no encontrado todavía")
        continue
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    if runs:
        run = runs[0]
        print(f"Último run: {run.info.run_id}")
        print(f"  Status  : {run.info.status}")
        for k, v in run.data.metrics.items():
            print(f"  {k:40s} {v:.4f}")

for model_name in ["xgb_fare", "xgb_duration"]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"⚠️  Modelo '{model_name}' no encontrado en el registry")
        continue
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"✅ {model_name} v{latest.version} → Production")

# Local paths (Jupyter container mounts project at /home/jovyan/work)
checks = {
    "Gold dist_media"      : f"{GOLD_LOCAL}/dist_media",
    "taxi_zone_lookup.csv" : f"{BRONZE_LOCAL}/taxi_zone_lookup.csv",
}

for label, path in checks.items():
    ok = os.path.exists(path)
    print(f"  {'✅' if ok else '❌'}  {label:30s}  {path}")

# MLflow model stages
for model_name in ["xgb_fare", "xgb_duration"]:
    versions = client.search_model_versions(f"name='{model_name}'")
    in_prod  = any(v.current_stage == "Production" for v in versions)
    print(f"  {'✅' if in_prod else '❌'}  MLflow model {model_name:20s}  stage=Production")

# Arrancar Streamlit (perfil 'app')
# Ejecutar desde el host (fuera del notebook) o descomenta si prefieres hacerlo aquí:

# result = subprocess.run(
#     ["docker", "compose", "--profile", "app", "up", "-d", "streamlit"],
#     capture_output=True, text=True, cwd="/home/jovyan/work"
# )
# print(result.stdout)

print("Para arrancar Streamlit, ejecuta en el host:")
print("  docker compose --profile app up -d streamlit")
print("  → http://localhost:8501")

# Arrancar el streaming job en background
# result = subprocess.run(
#     [
#         "docker", "exec", "-d", "spark-master",
#         "/opt/spark/bin/spark-submit", "--master", SPARK_MASTER,
#         "/opt/spark/src/etl/streaming_job.py",
#     ]
# )

# Simular datos: copiar un Parquet al hot-folder
import shutil, time

SRC = f"{BRONZE_PATH}/yellow_tripdata_2024-01.parquet"
DST = f"/opt/spark/data/streaming_source/batch_{int(time.time())}.parquet"

if os.path.exists(SRC):
    shutil.copy(SRC, DST)
    print(f"✅ Parquet copiado a streaming_source: {os.path.basename(DST)}")
else:
    print(f"⚠️  {SRC} no encontrado — usa otro Parquet de bronze")
