#!/usr/bin/env bash
# Descarga un ONNX de clasificacion MobileNet v2 para el pipeline demo (host Linux / Docker).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${DIR}/models/demo_mobilenet_v2/mobilenetv2-7.onnx"
URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"

mkdir -p "$(dirname "${OUT}")"
if [[ -f "${OUT}" ]]; then
  echo "Ya existe: ${OUT}"
  exit 0
fi
echo "Descargando ${URL}"
wget -q -O "${OUT}" "${URL}"
echo "Listo: ${OUT}"
