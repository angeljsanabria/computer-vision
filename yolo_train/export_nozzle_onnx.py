"""
Exporta best.pt del fine-tune nozzle a ONNX (misma receta que yolov8n COCO).

Receta (RKNN Toolkit 2.3.2): imgsz=640, opset=19.

Uso (desde la raiz del repo):
  python yolo_train/export_nozzle_onnx.py
  python yolo_train/export_nozzle_onnx.py --weights yolo_train/runs/detect/nozzle_v1/weights/best.pt

Salida:
  Yolo-Weights/yolov8n_nozzle.onnx

Siguiente paso: python yolo_train/exp_yolov8n_nozzle_rknn.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

RUN_NAME = "nozzle_v1"
DEFAULT_BEST_PT = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
ONNX_OUT = ROOT / "Yolo-Weights" / "yolov8n_nozzle.onnx"

IMGSZ = 640
ONNX_OPSET = 19


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX del modelo nozzle fine-tuned.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_BEST_PT,
        help="Ruta al best.pt tras train_nozzle.py.",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Tamano de entrada ONNX.")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset (<=19 para RKNN 2.3.2).")
    args = parser.parse_args()

    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(
            f"No se encuentra {weights}\n"
            "Entrena primero: python yolo_train/train_nozzle.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Instala ultralytics: pip install ultralytics") from e

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pesos:  {weights}")
    print(f"Export: imgsz={args.imgsz} opset={args.opset}")

    model = YOLO(str(weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
    )

    exported = Path(export_path).resolve()
    if exported != ONNX_OUT.resolve():
        if ONNX_OUT.is_file():
            ONNX_OUT.unlink()
        shutil.move(str(exported), str(ONNX_OUT))

    print(f"OK -> {ONNX_OUT}")
    print("Siguiente: python yolo_train/exp_yolov8n_nozzle_rknn.py")


if __name__ == "__main__":
    main()
