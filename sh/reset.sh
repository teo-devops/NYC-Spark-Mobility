#!/usr/bin/env bash
# sh/reset.sh
# ────────────────
# Reinicia el laboratorio por completo en un solo paso:
# 1. Apaga y limpia de forma segura todo el entorno sin requerir sudo (Teardown)
# 2. Re-inicializa el entorno (Bootstrap) y ejecuta opcionalmente el pipeline
#    según los argumentos que se le pasen.
#
# Uso: ./sh/reset.sh [--etl] [--train] [--stream]
#
# Flags:
#   --etl     Ejecuta el pipeline ETL Bronze→Silver→Gold tras levantar
#   --train   Ejecuta el entrenamiento del modelo tras el ETL
#   --stream  Arranca el job de Structured Streaming en background

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🔄 [RESET] Iniciando reinicio completo del laboratorio..."
echo "────────────────────────────────────────────────────────"

# 1. Ejecutar el teardown para limpiar todo de forma segura
./sh/teardown.sh

echo ""
echo "🚀 [RESET] Iniciando el nuevo entorno..."
echo "────────────────────────────────────────"

# 2. Ejecutar bootstrap pasando todos los argumentos recibidos
./sh/bootstrap.sh "$@"
