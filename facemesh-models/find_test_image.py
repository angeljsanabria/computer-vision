"""
Busca una imagen JPG/PNG usable para test FaceMesh en el repo o rutas conocidas.

Uso:
  python find_test_image.py
  python find_test_image.py --copy facemesh-models/test_face.jpg
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# Rutas tipicas (muchas estan gitignored en clones minimos)
_CANDIDATE_DIRS = (
    ROOT / "mobilenet_modelos" / "calib",
    ROOT / "facemesh-models" / "calib",
    ROOT / "9_retinaface_testing",
    ROOT / "Retinaface-Models",
    ROOT / "rknn-toolkit2" / "rknn-toolkit2" / "examples",
    ROOT / "rknn-toolkit2" / "rknn-toolkit-lite2" / "examples",
)


def _find_images() -> list[Path]:
    found: list[Path] = []
    for base in _CANDIDATE_DIRS:
        if not base.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            found.extend(sorted(base.rglob(ext)))
    # dedupe
    seen: set[str] = set()
    out: list[Path] = []
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--copy", type=Path, default=None, help="Copiar primera imagen encontrada")
    args = p.parse_args()

    images = _find_images()
    if not images:
        print("No hay JPG/PNG en rutas conocidas del repo.")
        print("Agrega una foto con cara en mobilenet_modelos/calib/ o pasa --image a test_onnx_inference.py")
        raise SystemExit(1)

    print("Imagenes encontradas:")
    for img in images[:20]:
        print(" ", img.relative_to(ROOT))
    if len(images) > 20:
        print(f"  ... y {len(images) - 20} mas")

    if args.copy:
        args.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[0], args.copy)
        print("copiado ->", args.copy)


if __name__ == "__main__":
    main()
