"""
Genera facemesh-models/calib/*.jpg 192x192 para cuantizacion INT8 del export RKNN.

Busca imagenes con caras en el repo (mobilenet calib, embeddings, camara_snap, etc.),
deduplica por MD5 y escribe dataset.txt.

Uso:
  cd facemesh-models
  py -3.10 prepare_calib.py
  py -3.10 prepare_calib.py --retinaface   # recorte con RetinaFace si hay onnxruntime
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_CALIB = ROOT / "calib"
DATASET_PATH = ROOT / "dataset.txt"
OUT_SIZE = 192

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path
    mode: str  # "resize" | "center_crop" | "retinaface"
    glob: str = "*"


# Fuentes aptas para calibracion FaceMesh (parches de cara ~192x192).
SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("mobilenet_calib", REPO / "mobilenet_modelos" / "calib", "resize"),
    SourceSpec(
        "embeddings_zero",
        REPO / "embeddings" / "faces_upd",
        "center_crop",
        glob="*_zero.jpg",
    ),
    SourceSpec("embeddings_faces", REPO / "embeddings" / "faces", "center_crop"),
    SourceSpec(
        "embeddings_pose",
        REPO / "embeddings" / "faces_upd",
        "center_crop",
        glob="*_der.jpg",
    ),
    SourceSpec(
        "embeddings_pose",
        REPO / "embeddings" / "faces_upd",
        "center_crop",
        glob="*_izq.jpg",
    ),
    SourceSpec(
        "camara_snap",
        REPO / "camara_snap",
        "center_crop",
        glob="latest_*retinaface*.jpg",
    ),
    SourceSpec(
        "camara_snap",
        REPO / "camara_snap",
        "center_crop",
        glob="latest_camara_snap.jpg",
    ),
    SourceSpec(
        "retinaface_result",
        REPO / "export_models",
        "center_crop",
        glob="result_retinaface_onnx.jpg",
    ),
    SourceSpec(
        "pfld_sample",
        REPO / "mesh-originals" / "pfld_106_face_landmarks",
        "center_crop",
        glob="1.png",
    ),
)


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_source_files(spec: SourceSpec) -> list[Path]:
    if not spec.path.is_dir():
        return []
    if spec.glob == "*":
        out: list[Path] = []
        for ext in IMG_EXTS:
            out.extend(sorted(spec.path.glob(f"*{ext}")))
            out.extend(sorted(spec.path.glob(f"*{ext.upper()}")))
        return out
    return sorted(spec.path.glob(spec.glob))


def _center_square_crop_bgr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _resize192(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)


def _patch_from_image(img: np.ndarray, mode: str, detector) -> np.ndarray | None:
    if img is None or img.size == 0:
        return None

    if mode == "retinaface" and detector is not None:
        sys.path.insert(0, str(REPO))
        from inference.retinaface.select_best import mejores_caras  # noqa: WPS433
        from inference.face_crop import crop_bbox_to_size  # noqa: WPS433
        from configs import settings as s  # noqa: WPS433

        dets = mejores_caras(detector.detect(img), top_n=1)
        if dets.has_faces:
            return crop_bbox_to_size(
                img,
                dets.dets[0],
                margin_frac=s.FACE_CROP_MARGIN_FRAC,
                out_size=OUT_SIZE,
            )

    if mode in ("resize", "retinaface"):
        h, w = img.shape[:2]
        if max(h, w) <= 512 and min(h, w) >= 64:
            return _resize192(img)

    cropped = _center_square_crop_bgr(img)
    return _resize192(cropped)


def _build_detector(use_retinaface: bool):
    if not use_retinaface:
        return None
    sys.path.insert(0, str(REPO))
    try:
        from inference import build_face_detector  # noqa: WPS433
        from configs import settings as s  # noqa: WPS433
    except ImportError as exc:
        print("WARN: no se pudo importar inference:", exc)
        return None

    onnx = REPO / "models_onnx" / "RetinaFace_mobile320.onnx"
    if not onnx.is_file():
        print("WARN: sin RetinaFace ONNX, fallback center_crop")
        return None
    try:
        det = build_face_detector(
            "pc",
            str(onnx),
            s.RETINAFACE_SCORE_DETECCION,
            s.RETINAFACE_SCORE_PRE_NMS,
        )
    except Exception as exc:  # noqa: BLE001
        print("WARN: RetinaFace no disponible:", exc)
        return None
    return det


def main() -> None:
    p = argparse.ArgumentParser(description="Preparar calib/ 192x192 para FaceMesh RKNN")
    p.add_argument(
        "--retinaface",
        action="store_true",
        help="Recortar con RetinaFace cuando sea posible (requiere onnxruntime)",
    )
    args = p.parse_args()

    detector = _build_detector(args.retinaface)
    OUT_CALIB.mkdir(parents=True, exist_ok=True)

    seen_hash: set[str] = set()
    seen_src: set[str] = set()
    lines: list[str] = []
    stats: dict[str, int] = {}

    for spec in SOURCE_SPECS:
        files = _iter_source_files(spec)
        if not files:
            print("SKIP (vacío):", spec.label, spec.path)
            continue

        mode = spec.mode
        if mode == "retinaface" or (args.retinaface and mode == "center_crop"):
            mode = "retinaface"

        for src in files:
            key = str(src.resolve())
            if key in seen_src:
                continue
            seen_src.add(key)

            img = cv2.imread(str(src))
            if img is None:
                print("WARN: no lee", src)
                continue

            patch = _patch_from_image(img, mode if spec.mode != "resize" else "resize", detector)
            if patch is None:
                print("WARN: sin parche", src.name)
                continue

            ok, buf = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                continue
            digest = _md5_bytes(buf.tobytes())
            if digest in seen_hash:
                print("SKIP dup", src.relative_to(REPO))
                continue
            seen_hash.add(digest)

            stem = f"{spec.label}_{src.stem}".replace(" ", "_")
            dst = OUT_CALIB / f"{stem}.jpg"
            dst.write_bytes(buf.tobytes())
            rel = f"calib/{dst.name}"
            lines.append(rel)
            stats[spec.label] = stats.get(spec.label, 0) + 1
            print("OK", rel, "<-", src.relative_to(REPO))

    if not lines:
        raise SystemExit(
            "No se genero ninguna imagen. Revisar rutas en SOURCE_SPECS o agregar JPG en "
            "mobilenet_modelos/calib/"
        )

    DATASET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\ndataset.txt ->", len(lines), "entradas")
    for label, count in sorted(stats.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
