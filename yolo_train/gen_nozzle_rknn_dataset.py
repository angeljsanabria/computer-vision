"""
Genera dataset.txt para calibracion INT8 del nozzle RKNN.

Toma imagenes de train/ y valid/ del dataset YOLO activo (nozzle_config.DATASET_ROOT)
y escribe rutas relativas a yolo_train/ en rknn_nozzle_dataset.txt.

Uso (desde la raiz del repo):
  python yolo_train/gen_nozzle_rknn_dataset.py
  python yolo_train/gen_nozzle_rknn_dataset.py --max-images 150
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUT = SCRIPT_DIR / "rknn_nozzle_dataset.txt"
DEFAULT_MAX = 120


def _collect_images(root: Path) -> list[Path]:
    found: list[Path] = []
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for path in sorted(img_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # RKNN dataset.txt parte paths en espacios; omitir nombres Roboflow con '(...)'.
            if any(ch in path.name for ch in (" ", "(", ")")):
                continue
            found.append(path.resolve())
    return found


def build_dataset_txt(
    *,
    dataset_root: Path,
    out_path: Path,
    max_images: int,
    seed: int,
) -> int:
    images = _collect_images(dataset_root)
    if not images:
        raise SystemExit(
            f"No hay imagenes en {dataset_root}/train|valid|test/images\n"
            "Revisa nozzle_config.DATASET_DIR."
        )

    rng = random.Random(seed)
    rng.shuffle(images)
    if max_images > 0:
        images = images[:max_images]

    root_anchor = nc.DATASET_ROOT.resolve().parent.resolve()
    # RKNN resuelve paths del dataset.txt respecto al directorio del .txt (yolo_train/).
    if out_path.parent.resolve() != SCRIPT_DIR.resolve():
        root_anchor = out_path.parent.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for path in images:
        try:
            rel = path.relative_to(root_anchor)
            lines.append(rel.as_posix())
        except ValueError:
            lines.append(path.as_posix())
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset de calibracion INT8 para exp_yolov8n_nozzle_rknn.py"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=nc.DATASET_ROOT,
        help="Carpeta YOLOv8 (train/valid/test).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Salida dataset.txt para RKNN build.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX,
        help="Max imagenes de calibracion (0 = todas).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla shuffle.")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"No existe dataset: {root}")

    n = build_dataset_txt(
        dataset_root=root,
        out_path=args.out.resolve(),
        max_images=args.max_images,
        seed=args.seed,
    )
    print(f"Version nozzle: {nc.NOZZLE_VERSION}")
    print(f"Dataset YOLO:     {root}")
    print(f"Imagenes:         {n}")
    print(f"OK -> {args.out.resolve()}")
    print("Ejecutar export desde la raiz del repo; paths relativos a yolo_train/.")
    print("Siguiente: python yolo_train/exp_yolov8n_nozzle_rknn.py")


if __name__ == "__main__":
    main()
