"""
Convierte labels YOLO-seg (poligono) a YOLO-detect (bbox AABB).

No modifica el export Roboflow original. Escribe:

  yolo_train/<DATASET_SRC>_detect/
    train|valid|test/images  (copias o hardlinks)
    train|valid|test/labels  (class cx cy w h)
    data.yaml

Uso (desde la raiz del repo):
  python yolo_train/prepare_nozzle_detect_labels.py
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

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Pares x,y normalizados -> cx,cy,w,h (clip [0,1])."""
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.0 or h <= 0.0:
        return None
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return cx, cy, w, h


def _line_to_detect(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        cls_id = int(float(parts[0]))
    except ValueError:
        return None
    nums = [float(x) for x in parts[1:]]
    # Ya es detect: class + 4 valores
    if len(nums) == 4:
        cx, cy, w, h = nums
        return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    box = _polygon_to_xywh(nums)
    if box is None:
        return None
    cx, cy, w, h = box
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_dataset(*, src_root: Path, dst_root: Path) -> tuple[int, int, int]:
    if not src_root.is_dir():
        raise SystemExit(f"No existe dataset fuente: {src_root}")

    n_images = 0
    n_labels = 0
    n_boxes = 0

    for split in SPLITS:
        src_img = src_root / split / "images"
        src_lbl = src_root / split / "labels"
        dst_img = dst_root / split / "images"
        dst_lbl = dst_root / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        if not src_img.is_dir():
            continue

        for img_path in sorted(src_img.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            _link_or_copy(img_path, dst_img / img_path.name)
            n_images += 1

            lbl_src = src_lbl / f"{img_path.stem}.txt"
            lbl_dst = dst_lbl / f"{img_path.stem}.txt"
            out_lines: list[str] = []
            if lbl_src.is_file():
                for raw in lbl_src.read_text(encoding="utf-8").splitlines():
                    converted = _line_to_detect(raw)
                    if converted is not None:
                        out_lines.append(converted)
                        n_boxes += 1
            lbl_dst.write_text(
                ("\n".join(out_lines) + "\n") if out_lines else "",
                encoding="utf-8",
            )
            n_labels += 1

    names_block = "\n".join(f"  - {name}" for name in nc.CLASS_NAMES)
    yaml_text = (
        f"path: {dst_root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {len(nc.CLASS_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    (dst_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return n_images, n_labels, n_boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poligono YOLO-seg -> bbox YOLO-detect (carpeta *_detect)."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=SCRIPT_DIR / nc.DATASET_SRC_DIR,
        help="Export Roboflow original (seg o detect).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=nc.DATASET_ROOT,
        help="Salida detect (default nozzle_config.DATASET_ROOT).",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if src == dst:
        raise SystemExit("src y dst no pueden ser la misma carpeta.")

    n_img, n_lbl, n_box = convert_dataset(src_root=src, dst_root=dst)
    print(f"Fuente:  {src}")
    print(f"Destino: {dst}")
    print(f"Imagenes: {n_img}  labels: {n_lbl}  boxes: {n_box}")
    print(f"Clases:   {list(nc.CLASS_NAMES)}")
    print("Siguiente: python yolo_train/train_nozzle.py")


if __name__ == "__main__":
    main()
