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
# Incluimos el perfil "app" para que streamlit-taxi se destruya junto con el resto;
# si no, sobrevive al down y queda con una referencia a una red ya eliminada,
# rompiendo el siguiente arranque con "network ... not found".
docker compose --profile app down -v --remove-orphans

echo "🧹 Limpiando directorios locales generados a través de Docker (sin requerir sudo)..."

# Usamos una imagen ligera de alpine montando la raíz del proyecto para borrar con permisos de root de forma no interactiva
docker run --rm -v "$ROOT:/workspace" alpine sh -c '
  echo "   - Eliminando capa Silver..."
  rm -rf /workspace/data/silver
  
  echo "   - Eliminando capa Gold..."
  rm -rf /workspace/data/gold
  
  echo "   - Eliminando hot-folder de Streaming..."
  rm -rf /workspace/data/streaming_source
  
  echo "   - Eliminando artefactos de MLflow..."
  rm -rf /workspace/mlflow_data
  
  echo "   - Eliminando eventos de Spark History Server..."
  rm -rf /workspace/spark-events
  
  echo "   - Eliminando caché local y archivos temporales..."
  rm -rf /workspace/.pytest_cache
  rm -rf /workspace/.idea
  rm -rf /workspace/app/.streamlit
  find /workspace -type d -name "__pycache__" -exec rm -rf {} +
'

echo "📂 Recreando directorios locales con permisos del usuario actual..."
# Recreamos la estructura de carpetas localmente. Al hacerlo fuera de Docker,
# las carpetas pertenecerán al usuario del host actual (evitando problemas de permisos/sudo)
for dir in data/silver data/gold data/streaming_source spark-events mlflow_data/artifacts secrets; do
  mkdir -p "$dir"
done

echo "✨ Teardown completo. El entorno está 100% limpio y listo."
echo "   (Tus archivos fuente originales en data/bronze se han mantenido a salvo)."
