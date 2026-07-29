"""
Fine-tune YOLOv8n (COCO preentrenado) para Bidon/Pico (nozzle_bidones).

Config central: yolo_train/nozzle_config.py (NOZZLE_VERSION, DATASET_DIR, IMGSZ).

Uso (desde la raiz del repo):
  python yolo_train/prepare_nozzle_detect_labels.py
  python yolo_train/train_nozzle.py
  python yolo_train/train_nozzle.py --epochs 50 --batch 8

Salida tipica:
  yolo_train/runs/detect/nozzle_bidones_v4/weights/best.pt

Siguiente paso: python yolo_train/export_nozzle_onnx.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

ROOT = nc.ROOT
PRETRAINED_PT = nc.PRETRAINED_PT
DATA_YAML = nc.DATA_YAML
DATASET_ROOT = nc.DATASET_ROOT
DATA_YAML_RESOLVED = nc.DATA_YAML_RESOLVED
RUN_NAME = nc.RUN_NAME

EPOCHS = 100
IMGSZ = nc.IMGSZ
BATCH = 16


def _validar_rutas() -> None:
    if not PRETRAINED_PT.is_file():
        raise SystemExit(f"No se encuentra el peso base: {PRETRAINED_PT}")
    if not DATA_YAML.is_file():
        raise SystemExit(f"No se encuentra data_nozzle.yaml: {DATA_YAML}")

    train_img = DATASET_ROOT / "train" / "images"
    val_img = DATASET_ROOT / "valid" / "images"
    if not train_img.is_dir() or not any(train_img.iterdir()):
        raise SystemExit(
            f"Sin imagenes de entrenamiento en: {train_img}\n"
            "Corre: python yolo_train/prepare_nozzle_detect_labels.py"
        )
    if not val_img.is_dir() or not any(val_img.iterdir()):
        raise SystemExit(f"Sin imagenes de validacion en: {val_img}")


def _escribir_data_yaml_resuelto() -> Path:
    path_str = DATASET_ROOT.resolve().as_posix()
    names_block = "\n".join(f"  - {name}" for name in nc.CLASS_NAMES)
    content = (
        f"path: {path_str}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {len(nc.CLASS_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    DATA_YAML_RESOLVED.write_text(content, encoding="utf-8")
    return DATA_YAML_RESOLVED


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n Bidon/Pico (nozzle_bidones)."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epocas de entrenamiento.")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Tamano de entrada (px).")
    parser.add_argument("--name", default=RUN_NAME, help="Nombre de la corrida (runs/detect/<name>).")
    args = parser.parse_args()

    _validar_rutas()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Instala ultralytics: pip install ultralytics") from e

    data_yaml = _escribir_data_yaml_resuelto()

    print(f"Version:   {nc.NOZZLE_VERSION}")
    print(f"Peso base: {PRETRAINED_PT}")
    print(f"Dataset:   {data_yaml}")
    print(f"Path:      {DATASET_ROOT.resolve()}")
    print(f"Clases:    {list(nc.CLASS_NAMES)}")
    print(f"Epocas:    {args.epochs}  batch: {args.batch}  imgsz: {args.imgsz}")

    model = YOLO(str(PRETRAINED_PT))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(SCRIPT_DIR / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
    )

    best_pt = SCRIPT_DIR / "runs" / "detect" / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        print(f"OK -> {best_pt}")
        print("Siguiente: python yolo_train/export_nozzle_onnx.py")
    else:
        print("Entrenamiento finalizado; revisa runs/detect/ por best.pt", file=sys.stderr)


if __name__ == "__main__":
    main()
