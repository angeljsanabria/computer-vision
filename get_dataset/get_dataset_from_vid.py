"""
Extrae frames espaciados de un video en get_dataset/vid/ para armar dataset de imagenes.

Lee un video (.mp4, .mov, .avi, .mkv, .webm), opcionalmente redimensiona,
ajusta brillo y guarda DATA_LEN capturas distribuidas a lo largo de todo el clip
en get_dataset/out/.

Si DATA_LEN supera la cantidad de frames del video, se limita a un frame por captura
disponible y se registra un warning.

Constantes al inicio del archivo (no van a settings.py).

Ejemplo:
  python get_dataset/get_dataset_from_vid.py
  python get_dataset/get_dataset_from_vid.py --video mi_clip.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# --- Salida de imagen ---
# Si True: redimensiona cada frame a OUT_WIDTH x OUT_HEIGHT antes de guardar.
# Si False: guarda el frame tal cual sale del video (resolucion nativa del clip).
FORCE_OUTPUT_SIZE = False
OUT_WIDTH = 640
OUT_HEIGHT = 420

# Brillo/contraste: dst = clip(alpha * src + beta, 0, 255). beta=0 sin cambio;
# valores tipicos para aclarar levemente footage oscuro: beta 10-30, alpha 1.0.
BRIGHTNESS_ALPHA = 1.0
BRIGHTNESS_BETA = 0

# Cantidad de imagenes objetivo (se acota a frames totales del video).
DATA_LEN = 70

# Prefijo de nombre: {IMAGE_NAME_PREFIX}_{count}.jpg
IMAGE_NAME_PREFIX = "nozzle_v_ctk_feria"
IMAGE_EXT = ".jpg"

# Video: vacio = primer archivo encontrado en vid/; si no, nombre de archivo concreto.
VIDEO_FILENAME = "vid_4.mov"

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = SCRIPT_DIR / "vid"
OUT_DIR = SCRIPT_DIR / "out"

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _list_videos() -> list[Path]:
    if not VIDEO_DIR.is_dir():
        return []
    return sorted(
        p
        for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def _resolve_video_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            path = VIDEO_DIR / explicit
        if not path.is_file():
            raise FileNotFoundError(f"No existe el video: {explicit}")
        return path.resolve()

    if VIDEO_FILENAME:
        path = VIDEO_DIR / VIDEO_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"VIDEO_FILENAME='{VIDEO_FILENAME}' no encontrado en {VIDEO_DIR}"
            )
        return path.resolve()

    candidates = _list_videos()
    if not candidates:
        raise FileNotFoundError(
            f"No hay videos en {VIDEO_DIR}. Coloca un .mp4/.mov/etc. o usa --video."
        )
    if len(candidates) > 1:
        logging.warning(
            "Varios videos en %s; usando el primero: %s",
            VIDEO_DIR,
            candidates[0].name,
        )
    return candidates[0].resolve()


def _frame_indices(total_frames: int, data_len: int) -> tuple[list[int], int]:
    """
    Indices de frame equiespaciados en [0, total_frames - 1].

    Retorna (indices, cantidad_efectiva). Si data_len > total_frames, effective = total.
    """
    if total_frames <= 0:
        return [], 0

    effective = min(max(1, data_len), total_frames)
    if data_len > total_frames:
        logging.warning(
            "DATA_LEN=%d > frames_totales=%d; se exportaran %d imagenes (una por frame max).",
            data_len,
            total_frames,
            effective,
        )

    if effective == 1:
        return [0], 1

    last = total_frames - 1
    indices = [int(round(i * last / (effective - 1))) for i in range(effective)]
    return indices, effective


def _adjust_brightness(frame_bgr: np.ndarray) -> np.ndarray:
    if BRIGHTNESS_ALPHA == 1.0 and BRIGHTNESS_BETA == 0:
        return frame_bgr
    return cv2.convertScaleAbs(frame_bgr, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)


def _resize_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if not FORCE_OUTPUT_SIZE:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w == OUT_WIDTH and h == OUT_HEIGHT:
        return frame_bgr
    return cv2.resize(frame_bgr, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA)


def _process_frame(frame_bgr: np.ndarray) -> np.ndarray:
    return _adjust_brightness(_resize_frame(frame_bgr))


def _read_frame_at(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


def _frame_count(cap: cv2.VideoCapture) -> int:
    """Frames totales; si OpenCV no los reporta (.mov), cuenta leyendo el clip."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        return total

    logging.warning(
        "FRAME_COUNT no disponible; contando frames (puede tardar en videos largos)."
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def extract_dataset(video_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total = _frame_count(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info(
        "Video: %s | frames=%d fps=%.2f size=%dx%d",
        video_path.name,
        total,
        fps,
        width,
        height,
    )

    indices, effective = _frame_indices(total, DATA_LEN)
    if not indices:
        cap.release()
        raise RuntimeError("El video no reporta frames (FRAME_COUNT=0).")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for count, frame_idx in enumerate(indices):
        frame = _read_frame_at(cap, frame_idx)
        if frame is None:
            logging.warning("[SKIP] frame %d no legible", frame_idx)
            continue

        out_img = _process_frame(frame)
        out_h, out_w = out_img.shape[:2]
        out_path = out_dir / f"{IMAGE_NAME_PREFIX}_{count}{IMAGE_EXT}"
        if not cv2.imwrite(str(out_path), out_img):
            logging.warning("[ERROR] no se pudo guardar: %s", out_path.name)
            continue

        saved += 1
        logging.info(
            "[OK] %s <- frame %d/%d (%dx%d)",
            out_path.name,
            frame_idx,
            max(total - 1, 0),
            out_w,
            out_h,
        )

    cap.release()
    logging.info(
        "Listo: %d/%d imagenes en %s (DATA_LEN pedido=%d, frames video=%d).",
        saved,
        effective,
        out_dir,
        DATA_LEN,
        total,
    )
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae frames espaciados de un video para dataset de imagenes.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Archivo en get_dataset/vid/ o ruta absoluta (default: primer video o VIDEO_FILENAME).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Carpeta de salida (default: {OUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()
    video_path = _resolve_video_path(args.video)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else OUT_DIR

    logging.info(
        "Config: out=%dx%d alpha=%.2f beta=%.0f DATA_LEN=%d prefix=%r",
        OUT_WIDTH,
        OUT_HEIGHT,
        BRIGHTNESS_ALPHA,
        BRIGHTNESS_BETA,
        DATA_LEN,
        IMAGE_NAME_PREFIX,
    )

    saved = extract_dataset(video_path, out_dir)
    if saved == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
