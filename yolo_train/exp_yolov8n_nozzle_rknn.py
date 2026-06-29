"""
Exporta Yolo-Weights/yolov8n_nozzle.onnx -> yolov8n_nozzle.rknn (target RK3568).

Copia de export_models/exp_yolov8n_rknn.py con rutas del modelo nozzle fine-tuned.
Misma config RKNN: mean 0, std 255, sin cuantizacion, platform rk3568.

Requisito: ONNX generado con export_nozzle_onnx.py (opset <= 19).

Uso en PC/WSL con rknn-toolkit2 2.3.2 (venv x86_64):
  python yolo_train/exp_yolov8n_nozzle_rknn.py

Salida:
  Yolo-Weights/yolov8n_nozzle.rknn

En la placa, copiar a rknn-toolkit-lite/ y apuntar RKNN_PATH en los scripts
de use_model_yolov8/ (CLASSES = ("fuel nozzle",)).
"""
from __future__ import annotations

from pathlib import Path

from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "Yolo-Weights" / "yolov8n_nozzle.onnx"
RKNN_PATH = ROOT / "Yolo-Weights" / "yolov8n_nozzle.rknn"

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
print("OK ->", RKNN_PATH)
print("Opcional en placa: cp Yolo-Weights/yolov8n_nozzle.rknn rknn-toolkit-lite/")
