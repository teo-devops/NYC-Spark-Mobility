# ⚡ NYC Taxi — Distributed Lab (Medallion Edition)

[![Spark](https://img.shields.io/badge/Spark-3.5.1-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11.3-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Integración profesional de `spark_ML` y `taxi-app-ai` bajo una arquitectura **Data Lakehouse (Medallion)**.  
Procesamiento batch + structured streaming + Spark ML/XGBoost + tracking con MLflow + app Streamlit.

---

## 🏗️ Arquitectura

```
NYC TLC Parquets
      │
      ▼
  [BRONZE]  ── StructType strict, carga raw
      │
      ▼  raw_to_silver.py + schemas.py
  [SILVER]  ── limpieza limpiar_con_logica_nyc(), join zonas, enriquecimiento
      │
      ▼  silver_to_gold.py
   [GOLD]   ── features ABT listas para modelo + tablas de streaming
      │
      ├──▶ train.py  ──▶ MLflow Server ──▶ modelo registrado
      │
      └──▶ Streamlit App  (carga modelo desde MLflow)

streaming_source/  ──▶  streaming_job.py  ──▶  Gold (micro-batch)
```

---

## 🚀 Quick Start

```bash
# 1. Clonar e iniciar
git clone <repo>
cd nyc-taxi-dist-lab
cp .env.template .env          # editar si hace falta

# 2. Descargar datos NYC TLC en data/bronze/
#    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#    yellow_tripdata_2025-03.parquet  (y los otros 3)
#    taxi_zone_lookup.csv

# 3. Levantar el cluster
./sh/bootstrap.sh

# 4. ETL batch  (desde Jupyter o directamente)
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/raw_to_silver.py

docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/silver_to_gold.py

# 5. Entrenar modelo
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/models/train.py

# 6. App Streamlit (fuera del cluster)
cd app && pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 URLs

| Servicio          | URL                        |
|-------------------|----------------------------|
| Jupyter Lab       | http://localhost:8888  (token: `spark123`) |
| Spark Master UI   | http://localhost:8080  |
| MLflow UI         | http://localhost:5000  |
| Spark History     | http://localhost:18080 |
| Streamlit App     | http://localhost:8501  |

---

## 📁 Estructura

```
nyc-taxi-dist-lab/
├── docker/
│   └── jupyter/Dockerfile       # imagen Jupyter con Spark + XGBoost + MLflow
├── data/
│   ├── bronze/                  # Parquets originales NYC TLC (git-ignored)
│   ├── silver/                  # Datos limpios Parquet
│   ├── gold/                    # Features ABT Parquet
│   └── streaming_source/        # Hot folder — mueve Parquets aquí para simular stream
├── src/
│   ├── common/
│   │   ├── schemas.py           # StructType para yellow, green y zones
│   │   └── utils.py             # haversine, SparkSession factory, logging helpers
│   ├── etl/
│   │   ├── raw_to_silver.py     # Batch ETL Bronze → Silver (con MLflow logging)
│   │   ├── silver_to_gold.py    # Feature engineering Silver → Gold
│   │   └── streaming_job.py     # Structured Streaming hot-folder → Gold
│   └── models/
│       ├── train.py             # SparkML Pipeline + XGBoost, registra en MLflow
│       └── predict.py           # Carga modelo MLflow, genera predicciones batch
├── app/
│   ├── app.py                   # Streamlit — carga modelo desde MLflow registry
│   └── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb             # EDA cuantitativo/cualitativo (de taxi-app-ai)
│   ├── 02_silver_exploration.ipynb
│   └── 03_model_experiments.ipynb
├── config/
│   ├── spark/spark-defaults.conf
│   └── hive/hive-site.xml
├── mlflow_artifacts/                 # Artefactos MLflow persistidos
├── sh/
│   └── bootstrap.sh
├── docker-compose.yml
├── .env.template
└── .gitignore
```

---

## 🔑 Variables de entorno (.env)

```env
SPARK_VERSION=3.5.1
SPARK_WORKER_MEMORY=2G
SPARK_WORKER_CORES=2
MLFLOW_VERSION=v2.11.3
MLFLOW_TRACKING_URI=http://mlflow:5000
JUPYTER_TOKEN=spark123
MARIADB_VERSION=10.11
MYSQL_ROOT_PASSWORD=rootpass
MYSQL_DATABASE=hive_metastore
MYSQL_USER=hive
MYSQL_PASSWORD=hivepass
HIVE_VERSION=4.0.0
GCP_KEY_LOCAL=./secrets/gcp-key.json
GCP_KEY_CONTAINER=/opt/spark/secrets/gcp-key.json
```

---

*Máster Big Data 2025/26 — Integración spark_ML + taxi-app-ai*
