"""
Exporta Yolo-Weights/yolov8n.onnx -> yolov8n.rknn (target RK3568).

Requisito RKNN Toolkit 2.3.2: ONNX con opset <= 19. Si exportaste con Ultralytics
y sale opset 20, regenerar:
  yolo export model=... pt format=onnx imgsz=640 opset=19

Funciona desde cualquier directorio de trabajo: rutas absolutas al repo.
"""
from __future__ import annotations

from pathlib import Path

from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "Yolo-Weights" / "yolov8n.onnx"
RKNN_PATH = ROOT / "Yolo-Weights" / "yolov8n.rknn"

rknn = RKNN(verbose=True)

# Ver RKNNToolKit2_API: target_platform toolkit2 incluye rk3566, rk3568, ...
# Placa MYD-LR3568: usar rk3568 (rk3566 es otro SoC; mismo ecosistema NPU pero nombre distinto).
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
