"""
Prepara fotos de referencia: faces/ -> faces_upd/.

Por cada imagen en embeddings/faces/:
  1. RetinaFace detecta la mejor cara y mide roll (linea entre ojos).
  2. Si |roll| > MAX_ABS_ROLL_DEG (25): warning, prefijo err_ en nombres y marca X roja.
  3. Si |roll| > MAX_TOLERANCE_ABS_ROLL_DEG (3): rota la imagen completa a 0 deg.
  4. Si |roll| <= 3: copia sin rotar.
  5. Crop bbox con margen segun flags ENABLE_* (_zero, _der, _izq, or/).
  6. Guarda recortes habilitados en embeddings/faces_upd/ y faces_upd/or/.

Ejemplo:
  python embeddings/prepare_faces_refs.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Constantes (solo este script; no van a settings.py)
MAX_ABS_ROLL_DEG = 25.0
MAX_TOLERANCE_ABS_ROLL_DEG = 3.0
APPLY_ROT_ABS_ROLL_DEG = 7.0
MIN_RETINAFACE_SCORE = 0.90

ENABLE_PROCESS_ROLL_ZERO = True
ENABLE_PROCESS_ROLL_DER = True
ENABLE_PROCESS_ROLL_IZQ = True
ENABLE_SAVE_CROP_ORIGINAL = True

_ROLL_EXCEEDED_COLOR_BGR = (0, 0, 255)
_ROLL_EXCEEDED_LINE_THICKNESS = 2

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
FACES_DIR = SCRIPT_DIR / "faces"
FACES_UPD_DIR = SCRIPT_DIR / "faces_upd"
FACES_OR_DIR = FACES_UPD_DIR / "or"


def _project_root(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        if (d / "configs" / "settings.py").is_file():
            return d
    raise RuntimeError("No se encontro configs/settings.py desde " + str(start_dir))


ROOT = _project_root(SCRIPT_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import settings as s  # noqa: E402
from inference import build_face_detector  # noqa: E402
from inference.face_align import landmarks_from_det_row  # noqa: E402
from inference.face_crop import bbox_crop_with_margin  # noqa: E402
from inference.face_pose import eye_roll_deg_from_det_row  # noqa: E402


@dataclass
class PrepareStats:
    ok: int = 0
    ok_corrected: int = 0
    ok_unchanged: int = 0
    warn_roll_exceeded: int = 0
    skip_score: int = 0
    skip_no_face: int = 0
    skip_read: int = 0
    errors: int = 0


def _validate_dirs() -> None:
    if not FACES_DIR.is_dir():
        raise SystemExit(f"No existe carpeta de entrada: {FACES_DIR}")
    if not FACES_UPD_DIR.is_dir():
        raise SystemExit(f"No existe carpeta de salida: {FACES_UPD_DIR}")
    if ENABLE_SAVE_CROP_ORIGINAL:
        FACES_OR_DIR.mkdir(parents=True, exist_ok=True)


def _any_output_enabled() -> bool:
    return (
        ENABLE_PROCESS_ROLL_ZERO
        or ENABLE_PROCESS_ROLL_DER
        or ENABLE_PROCESS_ROLL_IZQ
        or ENABLE_SAVE_CROP_ORIGINAL
    )


def _list_images(faces_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(faces_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]


def _best_det_row(dets: np.ndarray) -> np.ndarray | None:
    if dets.shape[0] == 0:
        return None
    return dets[np.argmax(dets[:, 4])]


def _rotate_frame_zero_roll(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Rota la imagen completa para horizontalizar la linea entre ojos (~0 deg)."""
    lmk = landmarks_from_det_row(det_row)
    roll_deg = eye_roll_deg_from_det_row(det_row)

    if abs(roll_deg) < 0.05:
        return frame_bgr, roll_deg

    left = lmk[0].astype(np.float32)
    right = lmk[1].astype(np.float32)
    eyes_center = ((left + right) * 0.5).astype(np.float32)
    h, w = frame_bgr.shape[:2]
    center = (float(eyes_center[0]), float(eyes_center[1]))
    M = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
    rotated = cv2.warpAffine(
        frame_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, roll_deg


def _rotate_frame_by_deg(frame_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota la imagen completa alrededor del centro (augmentacion desde 0 deg)."""
    h, w = frame_bgr.shape[:2]
    center = (w * 0.5, h * 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        frame_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _output_paths(img_path: Path, *, err: bool = False) -> tuple[Path, Path, Path, Path]:
    stem = img_path.stem
    suffix = img_path.suffix
    prefix = "err_" if err else ""
    or_path = FACES_OR_DIR / f"{prefix}{stem}{suffix}"
    zero_path = FACES_UPD_DIR / f"{prefix}{stem}_zero{suffix}"
    der_path = FACES_UPD_DIR / f"{prefix}{stem}_der{suffix}"
    izq_path = FACES_UPD_DIR / f"{prefix}{stem}_izq{suffix}"
    return or_path, zero_path, der_path, izq_path


def _save_bgr(path: Path, img_bgr: np.ndarray) -> bool:
    return bool(cv2.imwrite(str(path), img_bgr))


def _draw_roll_exceeded_mark(crop_bgr: np.ndarray) -> np.ndarray:
    """X roja de punta a punta: (0,0)-(max_x,max_y) y (0,max_y)-(max_x,0)."""
    marked = crop_bgr.copy()
    h, w = marked.shape[:2]
    max_x = w - 1
    max_y = h - 1
    thickness = max(
        _ROLL_EXCEEDED_LINE_THICKNESS,
        min(w, h) // 80,
    )
    cv2.line(
        marked,
        (0, 0),
        (max_x, max_y),
        _ROLL_EXCEEDED_COLOR_BGR,
        thickness,
    )
    cv2.line(
        marked,
        (0, max_y),
        (max_x, 0),
        _ROLL_EXCEEDED_COLOR_BGR,
        thickness,
    )
    return marked


def _crop_best_face(
    detector,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """RetinaFace sobre frame ya rotado -> recorte bbox (sin resize a 112)."""
    try:
        face_dets = detector.detect(frame_bgr)
    except Exception:
        return None

    row = _best_det_row(face_dets.dets)
    if row is None:
        return None

    score = float(row[4])
    if score < MIN_RETINAFACE_SCORE:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_crop_with_margin(
        row, w, h, s.FACE_CROP_MARGIN_FRAC
    )
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return None

    return crop, score


def _crop_from_det_row(frame_bgr: np.ndarray, det_row: np.ndarray) -> np.ndarray | None:
    """Recorte bbox desde una deteccion ya conocida (sin nueva inferencia)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_crop_with_margin(
        det_row, w, h, s.FACE_CROP_MARGIN_FRAC
    )
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return None
    return crop


def _save_original_crop(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    out_path: Path,
    *,
    det_score: float,
    img_name: str,
    mark_roll_exceeded: bool,
) -> bool:
    """Crop de la imagen original sin rotar (faces_upd/or/, sin sufijo _or)."""
    crop = _crop_from_det_row(frame_bgr, det_row)
    if crop is None:
        logging.warning("[ERROR] crop vacio (_or): %s", img_name)
        return False
    if mark_roll_exceeded:
        crop = _draw_roll_exceeded_mark(crop)
    if not _save_bgr(out_path, crop):
        logging.warning("[ERROR] no se pudo guardar: %s", out_path.name)
        return False
    ch, cw = crop.shape[:2]
    logging.info(
        "     original or/ -> %s crop=%dx%d score=%.3f",
        out_path.name,
        cw,
        ch,
        det_score,
    )
    return True


def _save_rotated_crop(
    detector,
    frame_bgr: np.ndarray,
    out_path: Path,
    *,
    variant_label: str,
    img_name: str,
    mark_roll_exceeded: bool,
) -> tuple[int, int] | None:
    """Detecta cara, crop bbox y guarda. Retorna (h, w) del crop o None si falla."""
    result = _crop_best_face(detector, frame_bgr)
    if result is None:
        logging.warning(
            "[ERROR] sin cara valida tras %s (crop): %s",
            variant_label,
            img_name,
        )
        return None

    crop, crop_score = result
    if mark_roll_exceeded:
        crop = _draw_roll_exceeded_mark(crop)
    if not _save_bgr(out_path, crop):
        logging.warning("[ERROR] no se pudo guardar: %s", out_path.name)
        return None

    ch, cw = crop.shape[:2]
    logging.info(
        "     %s -> %s crop=%dx%d score=%.3f",
        variant_label,
        out_path.name,
        cw,
        ch,
        crop_score,
    )
    return ch, cw


def _process_image(
    img_path: Path,
    *,
    detector,
    stats: PrepareStats,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning("[SKIP] no se pudo leer: %s", img_path.name)
        stats.skip_read += 1
        return

    try:
        face_dets = detector.detect(img)
    except Exception as exc:
        logging.warning("[ERROR] RetinaFace en %s: %s", img_path.name, exc)
        stats.errors += 1
        return

    row = _best_det_row(face_dets.dets)
    if row is None:
        logging.warning("[SKIP] sin cara: %s", img_path.name)
        stats.skip_no_face += 1
        return

    score = float(row[4])
    if score < MIN_RETINAFACE_SCORE:
        logging.warning(
            "[SKIP] score %.3f < min %.2f: %s",
            score,
            MIN_RETINAFACE_SCORE,
            img_path.name,
        )
        stats.skip_score += 1
        return

    roll_orig = eye_roll_deg_from_det_row(row)
    abs_roll = abs(roll_orig)
    mark_roll_exceeded = abs_roll > MAX_ABS_ROLL_DEG

    if mark_roll_exceeded:
        logging.warning(
            "[WARN] roll_orig=%.1f deg |roll| > max %.1f deg "
            "(se procesa con prefijo err_ y marca X roja): %s",
            roll_orig,
            MAX_ABS_ROLL_DEG,
            img_path.name,
        )
        stats.warn_roll_exceeded += 1

    need_zero_frame = (
        ENABLE_PROCESS_ROLL_ZERO
        or ENABLE_PROCESS_ROLL_DER
        or ENABLE_PROCESS_ROLL_IZQ
    )
    if need_zero_frame:
        if abs_roll > MAX_TOLERANCE_ABS_ROLL_DEG:
            out_img, _ = _rotate_frame_zero_roll(img, row)
            correction = "SI"
            stats.ok_corrected += 1
        else:
            out_img = img
            correction = "NO"
            stats.ok_unchanged += 1
    else:
        out_img = img
        correction = "N/A"

    or_path, zero_path, der_path, izq_path = _output_paths(
        img_path, err=mark_roll_exceeded
    )

    logging.info(
        "[OK] %s | roll_orig=%.1f deg | correccion=%s | score_inicial=%.3f%s",
        img_path.name,
        roll_orig,
        correction,
        score,
        " | err_+marca_X" if mark_roll_exceeded else "",
    )

    saved_any = False

    if ENABLE_SAVE_CROP_ORIGINAL:
        if not _save_original_crop(
            img,
            row,
            or_path,
            det_score=score,
            img_name=img_path.name,
            mark_roll_exceeded=mark_roll_exceeded,
        ):
            stats.errors += 1
            return
        saved_any = True

    variants: list[tuple[str, np.ndarray, Path]] = []
    if ENABLE_PROCESS_ROLL_ZERO:
        variants.append(("0 deg _zero", out_img, zero_path))
    if ENABLE_PROCESS_ROLL_DER:
        variants.append(
            (
                f"-{APPLY_ROT_ABS_ROLL_DEG:.0f} deg _der",
                _rotate_frame_by_deg(out_img, -APPLY_ROT_ABS_ROLL_DEG),
                der_path,
            )
        )
    if ENABLE_PROCESS_ROLL_IZQ:
        variants.append(
            (
                f"+{APPLY_ROT_ABS_ROLL_DEG:.0f} deg _izq",
                _rotate_frame_by_deg(out_img, APPLY_ROT_ABS_ROLL_DEG),
                izq_path,
            )
        )

    for variant_label, frame, out_path in variants:
        if _save_rotated_crop(
            detector,
            frame,
            out_path,
            variant_label=variant_label,
            img_name=img_path.name,
            mark_roll_exceeded=mark_roll_exceeded,
        ) is None:
            stats.errors += 1
            return
        saved_any = True

    if saved_any:
        stats.ok += 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    s.validar_todo()
    if s.INFERENCE_BACKEND == "none":
        raise SystemExit(
            "INFERENCE_BACKEND=none: usar pc o rk3568 (mismo backend que main_mov)."
        )

    if not _any_output_enabled():
        raise SystemExit(
            "Ninguna salida habilitada: activar al menos un ENABLE_PROCESS_* "
            "o ENABLE_SAVE_CROP_ORIGINAL."
        )

    _validate_dirs()

    images = _list_images(FACES_DIR)
    if not images:
        raise SystemExit(
            f"Sin imagenes en {FACES_DIR} (sufijos: {_IMAGE_SUFFIXES})"
        )

    logging.info("Entrada: %s (%d imagenes)", FACES_DIR.resolve(), len(images))
    logging.info("Salida enrolable: %s", FACES_UPD_DIR.resolve())
    if ENABLE_SAVE_CROP_ORIGINAL:
        logging.info("Salida referencia original: %s", FACES_OR_DIR.resolve())
    logging.info(
        "Salidas: zero=%s der=%s izq=%s original=%s",
        ENABLE_PROCESS_ROLL_ZERO,
        ENABLE_PROCESS_ROLL_DER,
        ENABLE_PROCESS_ROLL_IZQ,
        ENABLE_SAVE_CROP_ORIGINAL,
    )
    logging.info(
        "Umbrales: tolerancia=%.1f deg (corregir si |roll| >), max=%.1f deg (marca X si |roll| >), "
        "augment=+/-%.1f deg (_zero/_der/_izq), crop_margin=%.2f",
        MAX_TOLERANCE_ABS_ROLL_DEG,
        MAX_ABS_ROLL_DEG,
        APPLY_ROT_ABS_ROLL_DEG,
        s.FACE_CROP_MARGIN_FRAC,
    )
    logging.info("Backend: %s", s.INFERENCE_BACKEND)

    detector = build_face_detector()
    if detector is None:
        raise SystemExit("No se pudo crear detector.")

    stats = PrepareStats()
    try:
        for img_path in images:
            _process_image(img_path, detector=detector, stats=stats)
    finally:
        release = getattr(detector, "release", None)
        if callable(release):
            release()

    logging.info(
        "Resumen: %d imagenes | OK=%d (corregidas=%d sin_cambio=%d) "
        "roll_warn=%d score=%d sin_cara=%d lectura=%d error=%d",
        len(images),
        stats.ok,
        stats.ok_corrected,
        stats.ok_unchanged,
        stats.warn_roll_exceeded,
        stats.skip_score,
        stats.skip_no_face,
        stats.skip_read,
        stats.errors,
    )


if __name__ == "__main__":
    main()
