#!/usr/bin/env bash
# Crea arbol de despliegue en la placa (ejecutar en MYD como root o con sudo).
# Uso: sudo ./deploy_layout.sh demo_mobilenet_v2 0.1.0
set -euo pipefail
BASE="${EDGE_MODELS_ROOT:-/opt/edge-models}"
MODEL_ID="${1:?model_id}"
VERSION="${2:?version}"
TARGET="${BASE}/${MODEL_ID}/${VERSION}"
mkdir -p "${TARGET}"
echo "Directorio listo: ${TARGET}"
echo "Copia aqui model.rknn y manifest.json (scp/rsync)."
