"""
Constantes compartidas del pipeline nozzle YOLO.

Para una nueva version (v3):
  1. Export Roboflow -> yolo_train/nozzle_v3.yolov8 (o la carpeta que uses)
  2. Cambiar NOZZLE_VERSION y DATASET_DIR aca
  3. train_nozzle.py -> export_nozzle_onnx.py -> exp_yolov8n_nozzle_rknn.py

Los backups _v2 quedan en nozzle_v2.yolov8, yolov8n_nozzle_v2.onnx, etc.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# --- Cambiar para v3 ---
NOZZLE_VERSION = "v2"
DATASET_DIR = "nozzle.yolov8"  # dataset activo de entrenamiento
CLASS_NAME = "nozzle"

RUN_NAME = f"nozzle_{NOZZLE_VERSION}"

DATASET_ROOT = SCRIPT_DIR / DATASET_DIR
DATA_YAML = SCRIPT_DIR / "data_nozzle.yaml"
DATA_YAML_RESOLVED = SCRIPT_DIR / f"data_nozzle_{NOZZLE_VERSION}_resolved.yaml"

PRETRAINED_PT = ROOT / "Yolo-Weights" / "yolov8n.pt"
WEIGHTS_BEST_PT = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"

ONNX_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_nozzle_{NOZZLE_VERSION}.onnx"
RKNN_VERSIONED = ROOT / "Yolo-Weights" / f"yolov8n_nozzle_{NOZZLE_VERSION}.rknn"
# Alias "ultimo desplegado" (export lo actualiza; nozzle_yolo_v1.py lo usa por default)
ONNX_LATEST = ROOT / "Yolo-Weights" / "yolov8n_nozzle.onnx"
RKNN_LATEST = ROOT / "Yolo-Weights" / "yolov8n_nozzle.rknn"

# Backup congelado v2 (no tocar al entrenar v3)
DATASET_BACKUP_V2 = SCRIPT_DIR / "nozzle_v2.yolov8"
ONNX_BACKUP_V2 = ROOT / "Yolo-Weights" / "yolov8n_nozzle_v2.onnx"
WEIGHTS_BACKUP_V2 = ROOT / "Yolo-Weights" / "nozzle_v2_best.pt"
