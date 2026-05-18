#!/usr/bin/env bash
# sh/teardown.sh
# ────────────────
# Apaga el cluster y elimina de forma segura los volúmenes, carpetas generadas, 
# metadatos de bases de datos y artefactos de MLflow, dejando el proyecto
# completamente limpio (a excepción de los datos crudos originales en Bronze).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🛑 Apagando el cluster y eliminando contenedores, redes y volúmenes de Docker..."
docker compose down -v --remove-orphans

echo "🧹 Limpiando directorios locales generados (requiere sudo para archivos creados por Docker)..."

# Limpiamos capas de datos procesadas (manteniendo data/bronze intacta)
echo "   - Eliminando capa Silver: data/silver/"
sudo rm -rf data/silver/*
echo "   - Eliminando capa Gold: data/gold/"
sudo rm -rf data/gold/*
echo "   - Eliminando hot-folder de Streaming: data/streaming_source/"
sudo rm -rf data/streaming_source/*

# Limpiamos metadatos, artefactos y eventos
echo "   - Eliminando artefactos de MLflow: mlflow_data/"
sudo rm -rf mlflow_data/*
echo "   - Eliminando eventos de Spark History Server: spark-events/"
sudo rm -rf spark-events/*
echo "   - Eliminando cache local: .pytest_cache, __pycache__"
find . -type d -name "__pycache__" -exec sudo rm -rf {} +
sudo rm -rf .pytest_cache
sudo rm -rf .idea
sudo rm -rf app/.streamlit

echo "✨ Teardown completo. El entorno está 100% limpio."
echo "   (Tus archivos fuente originales en data/bronze se han mantenido a salvo)."
