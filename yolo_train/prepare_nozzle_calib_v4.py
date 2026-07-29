"""
Prepara imagenes de calibracion INT8/hybrid para nozzle_bidones RKNN v4.

Genera parches IMGSZ x IMGSZ stretch (mismo preproceso que inference/nozzle_bidon:
  stretch_bgr_to_rknn_input -> RGB uint8 NHWC con mean 0 / std 255 en RKNN).

Uso (desde la raiz del repo):
  python yolo_train/prepare_nozzle_calib_v4.py
  python yolo_train/prepare_nozzle_calib_v4.py --max-images 200
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JPEG_QUALITY = 92


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _stretch_input_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(
        frame_bgr,
        (nc.RKNN_INPUT_SIZE, nc.RKNN_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def _collect_yolo_images(root: Path) -> list[Path]:
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
            if any(ch in path.name for ch in (" ", "(", ")")):
                continue
            found.append(path.resolve())
    return found


def _collect_extra_sources(repo: Path) -> list[Path]:
    extras: list[Path] = []
    cam_dir = repo / "camara_snap"
    if cam_dir.is_dir():
        for pattern in ("latest_nozzle*.jpg", "latest_camara_snap*.jpg", "latest_*nozzle*.jpg"):
            extras.extend(sorted(cam_dir.glob(pattern)))
    return [p.resolve() for p in extras if p.is_file()]


def build_calib_v4(
    *,
    dataset_root: Path,
    out_dir: Path,
    dataset_txt: Path,
    max_images: int,
    seed: int,
) -> int:
    import random

    rng = random.Random(seed)
    sources = _collect_yolo_images(dataset_root) + _collect_extra_sources(nc.ROOT)
    if not sources:
        raise SystemExit(
            f"No hay imagenes en {dataset_root} ni en camara_snap/\n"
            "Revisa nozzle_config.DATASET_DIR / prepare_nozzle_detect_labels.py."
        )

    rng.shuffle(sources)
    if max_images > 0:
        sources = sources[:max_images]

    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hash: set[str] = set()
    lines: list[str] = []

    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            print("WARN: no lee", src)
            continue
        patch = _stretch_input_bgr(img)
        ok, buf = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        payload = buf.tobytes()
        digest = _md5_bytes(payload)
        if digest in seen_hash:
            continue
        seen_hash.add(digest)

        stem = src.stem.replace(" ", "_")[:80]
        dst_name = f"{stem}_{digest[:8]}.jpg"
        dst = out_dir / dst_name
        dst.write_bytes(payload)
        rel = dst.relative_to(SCRIPT_DIR).as_posix()
        lines.append(rel)
        print("OK", rel, "<-", src.relative_to(nc.ROOT) if src.is_relative_to(nc.ROOT) else src)

    if not lines:
        raise SystemExit("No se genero ninguna imagen de calibracion.")

    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Calib {nc.RKNN_INPUT_SIZE} stretch para RKNN nozzle_bidones v4."
    )
    parser.add_argument("--dataset-root", type=Path, default=nc.DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=nc.RKNN_CALIB_DIR)
    parser.add_argument("--dataset-txt", type=Path, default=nc.RKNN_CALIB_DATASET)
    parser.add_argument("--max-images", type=int, default=nc.RKNN_CALIB_MAX_IMAGES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = build_calib_v4(
        dataset_root=args.dataset_root.resolve(),
        out_dir=args.out_dir.resolve(),
        dataset_txt=args.dataset_txt.resolve(),
        max_images=args.max_images,
        seed=args.seed,
    )
    print(f"Imagenes calibracion: {n}")
    print(f"OK -> {args.out_dir.resolve()}")
    print(f"OK -> {args.dataset_txt.resolve()}")
    print("Siguiente: python yolo_train/exp_yolov8n_nozzle_rknn_v4.py")


if __name__ == "__main__":
    main()
