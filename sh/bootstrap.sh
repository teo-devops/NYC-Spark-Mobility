#!/usr/bin/env bash
# sh/bootstrap.sh
# ────────────────
# Levanta el Distributed Lab completo.
# Uso: ./sh/bootstrap.sh [--etl] [--train] [--stream]
# Flags:
#   --etl     Ejecuta el pipeline ETL Bronze→Silver→Gold tras levantar
#   --train   Ejecuta el entrenamiento del modelo tras el ETL
#   --stream  Arranca el job de Structured Streaming en background

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ETL=false
RUN_TRAIN=false
RUN_STREAM=false

for arg in "$@"; do
  case $arg in
    --etl)    RUN_ETL=true    ;;
    --train)  RUN_TRAIN=true  ;;
    --stream) RUN_STREAM=true ;;
  esac
done

# ── 1. Validar .env ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "⚠️  No existe .env — copiando desde .env.template"
  cp .env.template .env
fi

# ── 2. Crear carpetas del Data Lake si no existen ──────────────────────────────
for dir in data/bronze data/silver data/gold data/streaming_source spark-events mlflow_data/artifacts secrets; do
  mkdir -p "$dir"
done
echo "✅ Estructura de carpetas verificada"

# ── 3. Levantar cluster ────────────────────────────────────────────────────────
echo "🚀 Levantando cluster Docker Compose..."
docker compose up -d --build

# Esperar a que Spark Master esté listo
echo "⏳ Esperando a Spark Master (puede tardar ~30s)..."
until docker exec spark-master curl -sf http://localhost:8080 > /dev/null 2>&1; do
  sleep 3
done
echo "✅ Spark Master listo"

# Esperar a MLflow
echo "⏳ Esperando a MLflow..."
until curl -sf http://localhost:5000/health > /dev/null 2>&1; do
  sleep 3
done
echo "✅ MLflow listo"

# ── 4. ETL Pipeline ────────────────────────────────────────────────────────────
if [ "$RUN_ETL" = true ]; then
  echo ""
  echo "🔄 Ejecutando ETL Bronze → Silver..."
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.mlflow.trackingUri=http://mlflow:5000 \
    /opt/spark/src/etl/raw_to_silver.py

  echo "🔄 Ejecutando ETL Silver → Gold..."
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.mlflow.trackingUri=http://mlflow:5000 \
    /opt/spark/src/etl/silver_to_gold.py

  echo "✅ ETL completado"
fi

# ── 5. Entrenamiento ───────────────────────────────────────────────────────────
if [ "$RUN_TRAIN" = true ]; then
  echo ""
  echo "🏋️  Entrenando modelos..."
  docker exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.mlflow.trackingUri=http://mlflow:5000 \
    /opt/spark/src/models/train.py
  echo "✅ Entrenamiento completado"
fi

# ── 6. Streaming (background) ──────────────────────────────────────────────────
if [ "$RUN_STREAM" = true ]; then
  echo ""
  echo "📡 Arrancando Structured Streaming en background..."
  docker exec -d spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /opt/spark/src/etl/streaming_job.py
  echo "✅ Streaming job arrancado (background)"
  echo "   Simula datos: cp data/bronze/yellow_tripdata_2025-03.parquet data/streaming_source/batch_\$(date +%s).parquet"
fi

# ── 7. App Streamlit ────────────────────────────────────────────────────────────
echo ""
echo "🖥️  Arrancando app Streamlit..."
# --force-recreate evita heredar un contenedor de una sesión anterior que quedó
# apuntando a una red de Docker ya destruida (ver sh/teardown.sh).
docker compose --profile app up -d --build --force-recreate streamlit
echo "✅ Streamlit arrancado"

# ── URLs ───────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 🌐 NYC Taxi Distributed Lab — URLs"
echo "════════════════════════════════════════"
echo "  Jupyter Lab    → http://localhost:8888  (token: spark123)"
echo "  Spark UI       → http://localhost:8080"
echo "  MLflow UI      → http://localhost:5000"
echo "  Spark History  → http://localhost:18080"
echo "  Streamlit App  → http://localhost:8501"
echo "════════════════════════════════════════"
