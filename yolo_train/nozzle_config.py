"""
Constantes compartidas del pipeline nozzle YOLO.

Entrenamiento activo: v2 (nozzle.yolov8, best.pt en runs/detect/nozzle_v2/).

RKNN export v3: mismo ONNX v2, calibracion stretch 640 + hybrid INT8 (output0 FP16).
  prepare_nozzle_calib_v3.py -> exp_yolov8n_nozzle_rknn_v3.py

Los backups _v2 quedan en nozzle_v2.yolov8, yolov8n_nozzle_v2.onnx, etc.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# --- Entrenamiento / ONNX (v2 congelado) ---
NOZZLE_VERSION = "v2"
DATASET_DIR = "nozzle.yolov8"
CLASS_NAME = "nozzle"

RUN_NAME = f"nozzle_{NOZZLE_VERSION}"

DATASET_ROOT = SCRIPT_DIR / DATASET_DIR
DATA_YAML = SCRIPT_DIR / "data_nozzle.yaml"
DATA_YAML_RESOLVED = SCRIPT_DIR / f"data_nozzle_{NOZZLE_VERSION}_resolved.yaml"

PRETRAINED_PT = ROOT / "Yolo-Weights" / "yolov8n.pt"
WEIGHTS_BEST_PT = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"

ONNX_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_nozzle_{NOZZLE_VERSION}.onnx"
ONNX_LATEST = ROOT / "Yolo-Weights" / "yolov8n_nozzle.onnx"

# --- RKNN export v3 (hybrid INT8, despliegue placa) ---
RKNN_EXPORT_VERSION = "v3"
RKNN_INPUT_SIZE = 640
ONNX_RKNN_SOURCE = ONNX_VERSIONED
RKNN_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_nozzle_{RKNN_EXPORT_VERSION}.rknn"
RKNN_LATEST = ROOT / "Yolo-Weights" / "yolov8n_nozzle.rknn"
RKNN_DEPLOY = ROOT / "models" / f"yolov8n_nozzle_{RKNN_EXPORT_VERSION}.rknn"

RKNN_BUILD_DIR = SCRIPT_DIR / "rknn_build_v3"
RKNN_CALIB_DIR = SCRIPT_DIR / "rknn_calib_v3"
RKNN_CALIB_DATASET = SCRIPT_DIR / "rknn_nozzle_v3_dataset.txt"
RKNN_CALIB_MAX_IMAGES = 150

# Salida ONNX del grafo; hybrid quant la deja en FP16 (evita cls=0 en INT8 pleno).
# Fix documentado (ultralytics#23340 / mahdieh-jokar): AMBOS nodos, no solo output0.
#   custom_quantize_layers:
#     output0-rs: float16
#     output0: float16
RKNN_OUTPUT_NODE = "output0"
RKNN_HYBRID_FP16_NODES = ("output0-rs", "output0")

# Backup congelado v2
DATASET_BACKUP_V2 = SCRIPT_DIR / "nozzle_v2.yolov8"
ONNX_BACKUP_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.onnx"
WEIGHTS_BACKUP_V2 = ROOT / "Yolo-Weights" / "nozzle_v2_best.pt"

# Legacy (v2 INT8 roto; no usar)
RKNN_CALIB_DATASET_LEGACY = SCRIPT_DIR / "rknn_nozzle_dataset.txt"
RKNN_VERSIONED_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.rknn"
