"""
Constantes compartidas del pipeline nozzle / nozzle_bidones.

Entrenamiento activo: nozzle_bidones_v4
  dataset detect: picos_y_bidones.v1i.yolov8_detect (desde export Roboflow seg)
  imgsz 416, clases Bidon/Pico (IDs 0/1 = jerrycan/nozzle del export)

RKNN export v4: hybrid INT8 (output0 FP16), calib stretch 416.
  prepare_nozzle_calib_v4.py -> exp_yolov8n_nozzle_rknn_v4.py

Runtime placa: inference/nozzle_bidon/ (no mutar inference/nozzle v3@640).

Legacy v2/v3: backups en comentarios / paths _V2/_V3 abajo.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# --- Entrenamiento / ONNX (activo: nozzle_bidones_v4) ---
NOZZLE_VERSION = "nozzle_bidones_v4"
# Export Roboflow original (puede ser seg); no se pisa.
DATASET_SRC_DIR = "picos_y_bidones.v1i.yolov8"
# Dataset de trabajo YOLO-detect (bbox); lo genera prepare_nozzle_detect_labels.py
DATASET_DIR = "picos_y_bidones.v1i.yolov8_detect"
# Indices 0,1 = mismos IDs del export Roboflow (jerrycan, nozzle)
CLASS_NAMES = ("Bidon", "Pico")
# Compat scripts viejos que lean CLASS_NAME
CLASS_NAME = CLASS_NAMES[1]

RUN_NAME = NOZZLE_VERSION
IMGSZ = 416

DATASET_ROOT = SCRIPT_DIR / DATASET_DIR
DATA_YAML = SCRIPT_DIR / "data_nozzle.yaml"
DATA_YAML_RESOLVED = SCRIPT_DIR / f"data_nozzle_{NOZZLE_VERSION}_resolved.yaml"

PRETRAINED_PT = ROOT / "Yolo-Weights" / "yolov8n.pt"
WEIGHTS_BEST_PT = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"

# Un solo nombre versionado (sin alias duplicado).
ONNX_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_{NOZZLE_VERSION}.onnx"

# --- RKNN export v4 (hybrid INT8, imgsz 416) ---
RKNN_EXPORT_VERSION = "v4"
RKNN_INPUT_SIZE = IMGSZ
ONNX_RKNN_SOURCE = ONNX_VERSIONED
RKNN_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_{NOZZLE_VERSION}.rknn"
# Copia de deploy para la placa (mismo contenido que RKNN_VERSIONED).
RKNN_DEPLOY = ROOT / "models" / f"yolov8n_{NOZZLE_VERSION}.rknn"

RKNN_BUILD_DIR = SCRIPT_DIR / "rknn_build_v4"
RKNN_CALIB_DIR = SCRIPT_DIR / "rknn_calib_v4"
RKNN_CALIB_DATASET = SCRIPT_DIR / "rknn_nozzle_v4_dataset.txt"
RKNN_CALIB_MAX_IMAGES = 150

RKNN_OUTPUT_NODE = "output0"
RKNN_HYBRID_FP16_NODES = ("output0-rs", "output0")

# --- Legacy (no borrar; rollback demo v2/v3) ---
DATASET_BACKUP_V2 = SCRIPT_DIR / "nozzle_v2.yolov8"
ONNX_BACKUP_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.onnx"
WEIGHTS_BACKUP_V2 = ROOT / "Yolo-Weights" / "nozzle_v2_best.pt"
RKNN_CALIB_DATASET_LEGACY = SCRIPT_DIR / "rknn_nozzle_dataset.txt"
RKNN_VERSIONED_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.rknn"
RKNN_VERSIONED_V3 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v3.rknn"
RKNN_DEPLOY_V3 = ROOT / "models" / "yolov8n_nozzle_v3.rknn"
