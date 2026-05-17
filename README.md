# NYC Taxi — Distributed Lab (Medallion Edition)

[![Spark](https://img.shields.io/badge/Spark-3.5.1-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11.3-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Data Lakehouse distribuido sobre Apache Spark 3.5.1 con arquitectura Medallion (Bronze → Silver → Gold), entrenamiento con Spark ML, tracking con MLflow y app de predicción Streamlit.

---

## Arquitectura

```
NYC TLC Parquets  +  taxi_zone_lookup.csv
        │
        ▼  raw_to_silver.py
   [BRONZE]  ──→  [SILVER]   limpieza · normalización · join zonas · particionado fleet/month
                      │
                      ▼  silver_to_gold.py
                   [GOLD]
                    ├── features/     ABT para entrenamiento
                    ├── dist_media/   distancia media por par PU/DO  (→ Streamlit)
                    └── agg_hourly/   KPIs por hora  (→ EDA)
                      │
                      ├──▶  train.py  ──▶  MLflow Registry  (xgb_fare · xgb_duration)
                      │
                      └──▶  Streamlit App  carga modelos desde MLflow Production
                      
streaming_source/  ──▶  streaming_job.py  ──▶  gold/streaming/  (micro-batch 30s)
```

---

## Servicios

| Servicio | URL | Credenciales |
|---|---|---|
| Jupyter Lab | http://localhost:8888 | token: `spark123` |
| Spark Master UI | http://localhost:8080 | — |
| Spark History Server | http://localhost:18080 | — |
| MLflow UI | http://localhost:5000 | — |
| Streamlit App | http://localhost:8501 | solo tras entrenar |

---

## Quick Start

### 1. Preparar entorno

```bash
git clone <repo>
cd nyc-taxi-dist-lab
cp .env.template .env          # ajustar si hace falta
```

### 2. Descargar datos NYC TLC

Descarga los Parquets desde https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page y colócalos en `data/bronze/`:

```
data/bronze/
├── yellow_tripdata_2024-01.parquet
├── taxi_zone_lookup.csv
└── weather.csv                       # opcional
```

### 3. Levantar el cluster

```bash
./sh/bootstrap.sh
```

El script:
- Crea las carpetas del Data Lake si no existen
- Levanta todos los servicios con `docker compose up -d --build`
- Espera a que Spark Master y MLflow estén listos
- Imprime las URLs

Flags opcionales para ejecutar el pipeline automáticamente:

```bash
./sh/bootstrap.sh --etl            # Bootstrap + Bronze→Silver→Gold
./sh/bootstrap.sh --etl --train    # + entrenamiento de modelos
./sh/bootstrap.sh --etl --train --stream  # + streaming job en background
```

---

## Pipeline paso a paso

### Paso 1 — ETL Bronze → Silver

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/raw_to_silver.py
```

Produce `data/silver/` particionado por `fleet` y `month`. Registra métricas de calidad en MLflow (experimento `nyc_taxi_etl`).

### Paso 2 — Feature Engineering Silver → Gold

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/silver_to_gold.py
```

Produce tres tablas Gold:
- `data/gold/features/` — ABT con columnas `hour`, `day_of_week`, `month_num`, `is_weekend`, `fleet_int`
- `data/gold/dist_media/` — distancia media histórica por par PU/DO (mínimo 5 viajes)
- `data/gold/agg_hourly/` — KPIs por hora y flota

### Paso 3 — Entrenar modelos

```bash
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf "spark.driver.extraJavaOptions=-Xss8m" \
    --conf "spark.executor.extraJavaOptions=-Xss8m" \
    /opt/spark/src/models/train.py
```

Entrena dos pipelines `VectorAssembler → StandardScaler → GBTRegressor`:
- `xgb_fare` — predicción de tarifa
- `xgb_duration` — predicción de duración

Registra ambos modelos en MLflow Model Registry.

### Paso 4 — Promover modelos a Production

Desde MLflow UI (http://localhost:5000) o con el notebook `02_etl_to_streamlit_pipeline.ipynb`:

```python
import mlflow
client = mlflow.tracking.MlflowClient()
for model in ["xgb_fare", "xgb_duration"]:
    v = client.get_latest_versions(model)[0]
    client.transition_model_version_stage(model, v.version, "Production", archive_existing_versions=True)
```

### Paso 5 — Arrancar Streamlit

```bash
docker compose --profile app up -d streamlit
```

Abre http://localhost:8501. La app carga los modelos desde MLflow Registry (stage `Production`) y la tabla `dist_media` desde Gold para autocompletar la distancia según el par origen/destino.

---

## Structured Streaming (opcional)

```bash
# Arrancar el job en background
docker exec -d spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/streaming_job.py

# Simular llegada de datos (hot-folder)
cp data/bronze/yellow_tripdata_2024-01.parquet \
   data/streaming_source/batch_$(date +%s).parquet
```

El job procesa un archivo por micro-batch cada 30 segundos y escribe en `data/gold/streaming/` particionado por hora. El checkpoint se guarda en `data/gold/_checkpoints/streaming/`.

---

## Notebooks

| Notebook | Descripción |
|---|---|
| `notebooks/01_dta_corregido.ipynb` | EDA del dataset NYC TLC |
| `notebooks/02_etl_to_streamlit_pipeline.ipynb` | Orquestación y validación del pipeline completo |

---

## Variables de entorno (.env)

```env
SPARK_VERSION=3.5.1
SPARK_WORKER_MEMORY=2g
SPARK_WORKER_CORES=2

MLFLOW_VERSION=v2.11.3
MLFLOW_TRACKING_URI=http://mlflow:5000

JUPYTER_TOKEN=spark123

MARIADB_VERSION=10.11
HIVE_VERSION=4.0.0
MYSQL_ROOT_PASSWORD=rootpass
MYSQL_DATABASE=hive_metastore
MYSQL_USER=hive
MYSQL_PASSWORD=hivepass

MODEL_NAME_FARE=xgb_fare
MODEL_NAME_DURATION=xgb_duration
```

---

## Notas de arquitectura

- **Imagen Spark**: `apache/spark:3.5.1` (ASF oficial). Bitnami archivó sus tags versionados en sep 2025 — no revertir a `bitnami/spark`. La ruta base es `/opt/spark`, no `/opt/bitnami/spark`.
- **Imagen Jupyter**: `quay.io/jupyter/pyspark-notebook:spark-3.5.0` — incluye JupyterLab, PySpark, pandas, numpy, scikit-learn y matplotlib. Solo se añaden `mlflow`, `pymysql`, `xgboost`, `plotly`, `seaborn` y `streamlit`.
- **MLflow backend**: MariaDB en `hive-metastore-db`, base de datos `hive_metastore_mlflow` (creada por `config/db/init.sql`). Artefactos en `mlflow_data/artifacts/`.
- **YAML multi-línea**: Los `command:` de docker-compose usan formato lista con `|` literal block para evitar el problema de fold (`>`) que parte argumentos en tokens separados.
- **Filtros NYC TLC**: Los umbrales en `src/common/utils.py::FILTER_RULES` replican la lógica del notebook de análisis original. No cambiar sin revalidar el modelo.
- **git-ignored**: `data/`, `mlflow_data/`, `secrets/`, `spark-events/`, `.env`. Nunca commitear datos ni artefactos.

---

*Máster Big Data 2025/26 — NYC Taxi Distributed Lab*
