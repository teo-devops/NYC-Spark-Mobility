# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NYC Taxi Distributed Lab — a Data Lakehouse (Medallion architecture) built on Apache Spark 3.5.1. It processes NYC TLC taxi data through Bronze → Silver → Gold layers, trains regression models with Spark ML (GBTRegressor), tracks experiments with MLflow, and serves predictions via a Streamlit app.

## Commands

### Cluster Lifecycle

```bash
# Start everything (from repo root)
./sh/bootstrap.sh

# Start + run full pipeline
./sh/bootstrap.sh --etl --train --stream

# Start Streamlit app (requires trained models in MLflow Production stage)
docker compose --profile app up streamlit

# Tear down
docker compose down
```

### Spark Submit (run from repo root)

```bash
# Bronze → Silver ETL
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/raw_to_silver.py

# Silver → Gold feature engineering
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/silver_to_gold.py

# Train models (fare + duration)
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/models/train.py

# Start streaming job (background)
docker exec -d spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/src/etl/streaming_job.py

# Simulate streaming data arrival
cp data/bronze/yellow_tripdata_2024-01.parquet data/streaming_source/batch_$(date +%s).parquet
```

### Local Development

```bash
# Install Python deps (local notebooks / app)
pip install -r requirements.txt

# Run Streamlit app locally (after training)
cd app && pip install -r requirements.txt && streamlit run app.py

# Run tests
pytest test/
```

## Architecture

### Medallion Layers

Data flows through three Parquet layers under `data/`:

- **Bronze** (`data/bronze/`) — raw NYC TLC Parquets and `taxi_zone_lookup.csv`. Loaded with strict StructType (no schema inference).
- **Silver** (`data/silver/`) — cleaned, normalized, zone-enriched. Partitioned by `fleet` and `month`. Both Yellow and Green taxis share `SILVER_SCHEMA`.
- **Gold** (`data/gold/`) — three outputs:
  - `features/` — ABT (Analytics Base Table) for model training
  - `dist_media/` — historical average distance per PU/DO pair, used by Streamlit
  - `agg_hourly/` — KPI aggregations for dashboards
  - `streaming/` — output of the Structured Streaming job

### Services (docker-compose.yml)

| Service | Port | Role |
|---|---|---|
| `spark-master` | 8080, 7077 | Spark cluster master |
| `spark-worker-1/2` | — | Two workers, configured via `SPARK_WORKER_MEMORY` / `SPARK_WORKER_CORES` |
| `spark-history-server` | 18080 | Event log replay |
| `jupyter` | 8888 | JupyterLab (token: `spark123`) built on `docker/jupyter/Dockerfile` |
| `mlflow` | 5000 | Experiment tracking + Model Registry |
| `hive-metastore-db` | — | MariaDB backend for Hive + MLflow metadata |
| `hive-metastore` | — | External Hive Metastore for Spark SQL |
| `streamlit` | 8501 | Prediction app (profile: `app`) |

The `jar-downloader` service pre-fetches `mysql-connector-j.jar` into the `external-jars` volume on first startup.

**Image note:** Uses `apache/spark:3.5.1` (official ASF image). Bitnami archived its versioned tags in Sep 2025 — do not revert to `bitnami/spark`. The path convention is `/opt/spark`, not `/opt/bitnami/spark`.

### Source Code (`src/`)

All Spark jobs add `/opt/spark/src` to `sys.path` so they can import from `common/`.

- `common/schemas.py` — `YELLOW_SCHEMA`, `GREEN_SCHEMA`, `ZONE_SCHEMA`, `SILVER_SCHEMA` as Spark `StructType`. Always load raw Parquets with explicit schema.
- `common/utils.py` — `get_spark()` (SparkSession factory with Hive + MLflow + AQE config), `haversine_udf`, `apply_nyc_filters()`, `log_etl_metrics()`.
- `etl/raw_to_silver.py` — Batch job processing Yellow and Green datasets defined in `DATASETS` list.
- `etl/silver_to_gold.py` — Feature engineering: adds `hour`, `day_of_week`, `month_num`, `is_weekend`, `fleet_int`, `usd_per_km`, `min_per_km`.
- `etl/streaming_job.py` — Structured Streaming from `data/streaming_source/` hot-folder, micro-batch every 30s, one file per trigger.
- `models/train.py` — Trains two models (`xgb_fare`, `xgb_duration`) using `GBTRegressor` inside a `VectorAssembler → StandardScaler → GBTRegressor` Pipeline. Registers both in MLflow Model Registry.

### Streamlit App (`app/`)

Loads Spark ML models from MLflow Registry at stage `Production`. Uses Spark inside Streamlit for inference — requires a local SparkSession. Models must be promoted to `Production` stage in MLflow before the app will work.

### MLflow

- Backend store: MariaDB at `hive-metastore-db`, database `hive_metastore_mlflow` (created by `config/db/init.sql`).
- Artifact store: `mlflow_data/artifacts/` (bind-mounted into containers as `/mlflow`).
- Experiments: `nyc_taxi_etl` (ETL metrics) and `nyc_taxi_model` (training runs).
- Model names: `xgb_fare` and `xgb_duration`.

## Key Conventions

- `DATASETS` in `raw_to_silver.py` controls which fleet/month Parquets are processed — update this list when adding new data files.
- Filter thresholds (`FILTER_RULES` in `utils.py`) replicate the NYC TLC cleaning logic from the original notebook analysis and should not change without re-validating the model.
- The `predict.py` module in `src/models/` is a batch inference script (not used by the app); the app loads models directly via `mlflow.spark.load_model`.
- `data/bronze/`, `data/silver/`, `data/gold/`, `mlflow_data/`, `secrets/`, and `spark-events/` are git-ignored — never commit data or artifacts.
- `.env` is git-ignored; copy `.env.template` to `.env` before first run.
