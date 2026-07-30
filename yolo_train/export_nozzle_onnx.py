"""
Exporta best.pt del fine-tune nozzle_bidones a ONNX.

Config: yolo_train/nozzle_config.py
Receta (RKNN Toolkit 2.3.2): imgsz desde nc.IMGSZ (416), opset=19.

Uso (desde la raiz del repo):
  python yolo_train/export_nozzle_onnx.py
  python yolo_train/export_nozzle_onnx.py --weights yolo_train/runs/detect/nozzle_bidones_v8/weights/best.pt

Salida:
  Yolo-Weights/yolov8n_nozzle_bidones_v8.onnx

Siguiente paso: python yolo_train/prepare_nozzle_calib_v8.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

DEFAULT_BEST_PT = nc.WEIGHTS_BEST_PT
ONNX_OUT = nc.ONNX_VERSIONED

IMGSZ = nc.IMGSZ
ONNX_OPSET = 19


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX nozzle_bidones fine-tuned.")
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

    print(f"Version: {nc.NOZZLE_VERSION}")
    print(f"Pesos:   {weights}")
    print(f"Export:  imgsz={args.imgsz} opset={args.opset}")

    model = YOLO(str(weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
    )

    exported = Path(export_path).resolve()
    if exported.resolve() != ONNX_OUT.resolve():
        if ONNX_OUT.is_file():
            ONNX_OUT.unlink()
        shutil.copy2(str(exported), str(ONNX_OUT))
        if exported.exists():
            exported.unlink()

    print(f"OK -> {ONNX_OUT}")
    print("Siguiente: python yolo_train/prepare_nozzle_calib_v8.py")


if __name__ == "__main__":
    main()
