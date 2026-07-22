"""
Exporta Yolo-Weights/yolov8n_nozzle_<version>.onnx -> yolov8n_nozzle_<version>.rknn (RK3568).

Config: yolo_train/nozzle_config.py

Uso en PC/WSL con rknn-toolkit2 2.3.2 (venv x86_64):
  python yolo_train/exp_yolov8n_nozzle_rknn.py

Salida:
  Yolo-Weights/yolov8n_nozzle_v2.rknn  (versionado)
  Yolo-Weights/yolov8n_nozzle.rknn     (alias ultimo desplegado)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

from rknn.api import RKNN

ONNX_PATH = nc.ONNX_VERSIONED
RKNN_PATH = nc.RKNN_VERSIONED
RKNN_LATEST = nc.RKNN_LATEST

if not ONNX_PATH.is_file():
    raise SystemExit(
        f"No se encuentra {ONNX_PATH}\n"
        "Exporta antes: python yolo_train/export_nozzle_onnx.py"
    )

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3568",
)

print(f"Version: {nc.NOZZLE_VERSION}")
print("--> load_onnx")
print("    ", ONNX_PATH)
ret = rknn.load_onnx(model=str(ONNX_PATH))
if ret != 0:
    raise SystemExit(f"load_onnx failed: {ret}")

print("--> build")
ret = rknn.build(do_quantization=False)
if ret != 0:
    raise SystemExit(f"build failed: {ret}")

print("--> export_rknn")
ret = rknn.export_rknn(str(RKNN_PATH))
if ret != 0:
    raise SystemExit(f"export failed: {ret}")

rknn.release()
shutil.copy2(str(RKNN_PATH), str(RKNN_LATEST))
print("OK ->", RKNN_PATH)
print("OK ->", RKNN_LATEST, "(alias desplegado)")
