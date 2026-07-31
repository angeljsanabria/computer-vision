"""
Prueba en PC del modelo fuel nozzle exportado a ONNX.

Config: yolo_train/nozzle_config.py

Uso (desde la raiz del repo, despues de export_nozzle_onnx.py):
  python yolo_train/detect_nozzle_onnx_img.py
  python yolo_train/detect_nozzle_onnx_img.py --img ruta/a/foto.jpg
  python yolo_train/detect_nozzle_onnx_img.py --conf 0.4

Requiere: ultralytics, opencv-python.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

ONNX_PATH = nc.ONNX_VERSIONED
DEFAULT_IMG = nc.DATASET_ROOT / "valid" / "images"

CONF_MIN = 0.25
DELAY_SEC = 5


def _resolver_imagen(img_arg: str | None) -> Path:
    if img_arg:
        p = Path(img_arg)
        if not p.is_file():
            raise SystemExit(f"No se encuentra la imagen: {p}")
        return p.resolve()

    valid_dir = DEFAULT_IMG
    if not valid_dir.is_dir():
        raise SystemExit(
            f"No hay carpeta valid/images en el dataset: {valid_dir}\n"
            "Pasa una imagen con --img ruta/a/foto.jpg"
        )
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        candidatas = sorted(valid_dir.glob(ext))
        if candidatas:
            return candidatas[0].resolve()
    raise SystemExit(f"Sin imagenes en {valid_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect fuel nozzle con ONNX en PC.")
    parser.add_argument("--img", help="Imagen a probar (default: primera de valid/images).")
    parser.add_argument("--conf", type=float, default=CONF_MIN, help="Umbral de confianza.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ONNX_PATH,
        help="Ruta al .onnx (default: Yolo-Weights/yolov8n_nozzle.onnx).",
    )
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(
            f"No se encuentra el modelo: {onnx_path}\n"
            "Exporta antes: python yolo_train/export_nozzle_onnx.py"
        )

    img_path = _resolver_imagen(args.img)

    model = YOLO(str(onnx_path))
    print(f"Modelo:  {onnx_path.name}")
    print(f"Clases:  {model.names}")
    print(f"Imagen:  {img_path}")
    print(f"Conf:    {args.conf}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"No se pudo leer la imagen: {img_path}")

    results = model(frame, conf=args.conf, imgsz=nc.IMGSZ, verbose=False)
    det = results[0].boxes
    n = 0 if det is None else len(det)
    print(f"Detecciones: {n}")
    if n > 0:
        for i, box in enumerate(det):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].tolist()
            print(f"  [{i}] {name} conf={conf:.2f} bbox={[round(v, 1) for v in xyxy]}")

    annotated = results[0].plot()
    win = "Fuel nozzle ONNX"
    cv2.imshow(win, annotated)
    cv2.waitKey(1)
    time.sleep(DELAY_SEC)
    print("Cualquier tecla para cerrar")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
