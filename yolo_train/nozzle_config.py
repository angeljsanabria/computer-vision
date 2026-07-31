"""
Constantes compartidas del pipeline nozzle / nozzle_bidones.

Entrenamiento activo: nozzle_bidones_v12
  dataset Roboflow: picos-bidones-v12 (labels poligono en export)
  dataset detect: picos-bidones-v12_detect (bbox via prepare_nozzle_detect_labels)
  imgsz 640, clases Bidon/Pico (IDs 0/1 = Bidon/nozzle del export)

RKNN export v12: hybrid INT8 (output0 FP16), calib stretch 640.
  prepare_nozzle_calib_v8.py -> exp_yolov8n_nozzle_rknn_v8.py

Runtime placa: inference/nozzle_bidon/ (no mutar inference/nozzle v3@640).

Legacy: v2/v3/v4/v7/v8 paths abajo.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# --- Entrenamiento / ONNX (activo: nozzle_bidones_v12) ---
NOZZLE_VERSION = "nozzle_bidones_v12"
# Export Roboflow original (poligono/seg); no se pisa.
DATASET_SRC_DIR = "picos-bidones-v12"
# Dataset de trabajo YOLO-detect (bbox); lo genera prepare_nozzle_detect_labels.py
DATASET_DIR = "picos-bidones-v12_detect"
# Indices 0,1 = mismos IDs del export Roboflow (Bidon, nozzle)
CLASS_NAMES = ("Bidon", "Pico")
# Compat scripts viejos que lean CLASS_NAME
CLASS_NAME = CLASS_NAMES[1]

RUN_NAME = NOZZLE_VERSION
IMGSZ = 640

DATASET_ROOT = SCRIPT_DIR / DATASET_DIR
DATA_YAML = SCRIPT_DIR / "data_nozzle.yaml"
DATA_YAML_RESOLVED = SCRIPT_DIR / f"data_nozzle_{NOZZLE_VERSION}_resolved.yaml"

PRETRAINED_PT = ROOT / "Yolo-Weights" / "yolov8n.pt"
WEIGHTS_BEST_PT = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"

# Un solo nombre versionado (sin alias duplicado).
ONNX_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_{NOZZLE_VERSION}.onnx"

# --- RKNN export v12 (hybrid INT8, imgsz 640) ---
RKNN_EXPORT_VERSION = "v12"
RKNN_INPUT_SIZE = IMGSZ
ONNX_RKNN_SOURCE = ONNX_VERSIONED
RKNN_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_{NOZZLE_VERSION}.rknn"
# Copia de deploy para la placa (mismo contenido que RKNN_VERSIONED).
RKNN_DEPLOY = ROOT / "models" / f"yolov8n_{NOZZLE_VERSION}.rknn"

RKNN_BUILD_DIR = SCRIPT_DIR / "rknn_build_v12"
RKNN_CALIB_DIR = SCRIPT_DIR / "rknn_calib_v12"
RKNN_CALIB_DATASET = SCRIPT_DIR / "rknn_nozzle_v12_dataset.txt"
RKNN_CALIB_MAX_IMAGES = 150

RKNN_OUTPUT_NODE = "output0"
RKNN_HYBRID_FP16_NODES = ("output0-rs", "output0")

# --- Legacy (no borrar; rollback) ---
DATASET_BACKUP_V2 = SCRIPT_DIR / "nozzle_v2.yolov8"
ONNX_BACKUP_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.onnx"
WEIGHTS_BACKUP_V2 = ROOT / "Yolo-Weights" / "nozzle_v2_best.pt"
RKNN_CALIB_DATASET_LEGACY = SCRIPT_DIR / "rknn_nozzle_dataset.txt"
RKNN_VERSIONED_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.rknn"
RKNN_VERSIONED_V3 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v3.rknn"
RKNN_DEPLOY_V3 = ROOT / "models" / "yolov8n_nozzle_v3.rknn"
RKNN_VERSIONED_V4 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_bidones_v4.rknn"
RKNN_DEPLOY_V4 = ROOT / "models" / "yolov8n_nozzle_bidones_v4.rknn"
RKNN_VERSIONED_V7 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_bidones_v7.rknn"
RKNN_DEPLOY_V7 = ROOT / "models" / "yolov8n_nozzle_bidones_v7.rknn"
RKNN_VERSIONED_V8 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_bidones_v8.rknn"
RKNN_DEPLOY_V8 = ROOT / "models" / "yolov8n_nozzle_bidones_v8.rknn"
RKNN_VERSIONED_V11 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_bidones_v11.rknn"
RKNN_DEPLOY_V11 = ROOT / "models" / "yolov8n_nozzle_bidones_v11.rknn"
