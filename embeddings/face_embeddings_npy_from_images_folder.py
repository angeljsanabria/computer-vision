"""
Enrolamiento batch: embeddings/faces_upd/ -> gallery.npy + gallery_meta.json.

Solo crop bbox (sin align ni roll-fix). Mide roll de ojos; si |roll| supera el umbral,
avisa y omite la foto (no corrige la imagen).

Entrada fija: embeddings/faces_upd/ (patron {id}_{nombre}[_zero|_der|_izq].ext).
Archivos con prefijo err_ (error de prepare) y referencias _or no se enrolan.
Salida fija: gallery.npy (N, 128) L2-normalizado por fila y gallery_meta.json
en el directorio del script.

Usa build_face_detector / build_embedder (configs INFERENCE_BACKEND) y crop via
prepare_face_patch (margen FACE_CROP_MARGIN_FRAC en settings).

Ejemplo:
  python embeddings/face_embeddings_npy_from_images_folder.py
  python embeddings/face_embeddings_npy_from_images_folder.py --flip-avg
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Constantes de enrolamiento (solo este script; no van a settings.py)
MAX_ABS_ROLL_DEG = 44.0
MIN_RETINAFACE_SCORE = 0.90
MAX_ID_DIGITS = 10
GALLERY_VERSION = 1
GALLERY_PREPROCESS = "crop"
GALLERY_DATA_NORMALIZADA = 1

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FILENAME_RE = re.compile(
    r"^(\d+)_(.+?)(?:_(izq|der|zero))?\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)
_ROT_TAG_TO_LABEL = {
    "izq": "izquierda",
    "der": "derecha",
    "zero": "cero",
}
_ROT_DEFAULT_LABEL = "cero"
_ERR_PREFIX = "err_"

SCRIPT_DIR = Path(__file__).resolve().parent
FACES_DIR = SCRIPT_DIR / "faces_upd"
GALLERY_NPY = SCRIPT_DIR / "gallery.npy"
GALLERY_META_JSON = SCRIPT_DIR / "gallery_meta.json"


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
from inference import build_embedder, build_face_detector  # noqa: E402
from inference.face_pose import eye_roll_deg_from_det_row  # noqa: E402
from inference.face_preprocess import prepare_face_patch  # noqa: E402
from inference.mobilefacenet.constants import EMBED_DIM  # noqa: E402
from inference.mobilefacenet.norm import l2_normalize  # noqa: E402


@dataclass
class EnrollStats:
    ok: int = 0
    skip_invalid_name: int = 0
    skip_err_ref: int = 0
    skip_or_ref: int = 0
    skip_roll: int = 0
    skip_score: int = 0
    skip_no_face: int = 0
    skip_read: int = 0
    errors: int = 0


@dataclass
class EnrollOk:
    vec: np.ndarray
    entry: dict[str, Any]


@dataclass
class GalleryBuild:
    vectors: list[np.ndarray] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)


def _list_images(faces_dir: Path) -> tuple[list[Path], int]:
    """Lista imagenes enrolables; retorna (paths, cantidad descartadas err_)."""
    enrolable: list[Path] = []
    skip_err = 0
    for p in sorted(faces_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if p.name.startswith(_ERR_PREFIX):
            skip_err += 1
            logging.warning(
                "[SKIP] imagen con error (prefijo err_), no enrolar: %s",
                p.name,
            )
            continue
        enrolable.append(p)
    return enrolable, skip_err


def _is_or_reference(img_path: Path) -> bool:
    """Referencia visual de prepare (sufijo _or); no entra a la galeria."""
    return bool(re.search(r"_or\.(jpg|jpeg|png|bmp|webp)$", img_path.name, re.IGNORECASE))


def _parse_face_filename(img_path: Path) -> tuple[str, str, str] | None:
    """
    Retorna (person_id zfill(10), nombre, rotacion) o None si el nombre no cumple.

    rotacion: "cero" (sin sufijo o _zero), "izquierda" (_izq), "derecha" (_der).
    Prefijo err_ y sufijo _or no se enrolan.
    """
    m = _FILENAME_RE.match(img_path.name)
    if m is None:
        return None

    id_raw, name_part, rot_tag, _ext = m.group(1), m.group(2), m.group(3), m.group(4)
    if len(id_raw) > MAX_ID_DIGITS:
        logging.warning(
            "[SKIP] id con mas de %d digitos (%s): %s",
            MAX_ID_DIGITS,
            id_raw,
            img_path.name,
        )
        return None

    person_id = id_raw.zfill(MAX_ID_DIGITS)
    nombre = name_part.replace("-", " ")

    if rot_tag is None:
        rotacion = _ROT_DEFAULT_LABEL
    else:
        rotacion = _ROT_TAG_TO_LABEL.get(rot_tag.lower())
        if rotacion is None:
            logging.warning("[SKIP] sufijo rotacion invalido: %s", img_path.name)
            return None

    return person_id, nombre, rotacion


def _rel_img_path(img_path: Path) -> str:
    return f"{FACES_DIR.name}/{img_path.name}"


def _release_inference(*objs) -> None:
    for obj in objs:
        release = getattr(obj, "release", None)
        if callable(release):
            release()


def _log_enrollment_config(*, flip_avg: bool) -> None:
    logging.info("Entrada: %s", FACES_DIR.resolve())
    logging.info("Salida: %s, %s", GALLERY_NPY.resolve(), GALLERY_META_JSON.resolve())
    logging.info(
        "Umbrales: score>=%.2f, |roll|<=%.1f deg, flip_avg=%s",
        MIN_RETINAFACE_SCORE,
        MAX_ABS_ROLL_DEG,
        flip_avg,
    )
    logging.info("Backend: %s", s.INFERENCE_BACKEND)


def _empty_deteccion() -> dict[str, Any]:
    return {
        "count": 0,
        "last_seen": None,
        "autorizaciones_validas": 0,
        "autorizaciones_denegadas": 0,
    }


def _best_det_row(dets: np.ndarray) -> np.ndarray | None:
    if dets.shape[0] == 0:
        return None
    return dets[np.argmax(dets[:, 4])]


def _process_one_image(
    img_path: Path,
    *,
    person_id: str,
    nombre: str,
    rotacion: str,
    detector,
    embedder,
    flip_avg: bool,
) -> EnrollOk | str:
    """
    Procesa una imagen. Retorna EnrollOk o: skip_roll | skip_score | skip_no_face |
    skip_read | error
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning("[SKIP] no se pudo leer: %s", img_path)
        return "skip_read"

    try:
        face_dets = detector.detect(img)
    except Exception as exc:
        logging.warning("[ERROR] RetinaFace en %s: %s", img_path.name, exc)
        return "error"

    row = _best_det_row(face_dets.dets)
    if row is None:
        logging.warning("[SKIP] sin cara: %s", img_path.name)
        return "skip_no_face"

    score = float(row[4])
    if score < MIN_RETINAFACE_SCORE:
        logging.warning(
            "[SKIP] score %.3f < min %.2f: %s",
            score,
            MIN_RETINAFACE_SCORE,
            img_path.name,
        )
        return "skip_score"

    roll_deg = eye_roll_deg_from_det_row(row)
    if abs(roll_deg) > MAX_ABS_ROLL_DEG:
        logging.warning(
            "[SKIP] foto no procesable: |roll|=%.1f deg > max %.1f deg "
            "(mirar de frente): %s",
            abs(roll_deg),
            MAX_ABS_ROLL_DEG,
            img_path.name,
        )
        return "skip_roll"

    try:
        patch = prepare_face_patch(
            img,
            row,
            arcface_align_enable=False,
            rot_align_simple_enable=False,
            max_abs_roll_deg=s.FACE_ROLL_MAX_DEG,
            crop_margin_frac=s.FACE_CROP_MARGIN_FRAC,
        )
        vec = embedder.embed(patch.bgr)
        if flip_avg:
            flip_bgr = cv2.flip(patch.bgr, 1)
            vec2 = embedder.embed(flip_bgr)
            vec = l2_normalize(vec + vec2)
        else:
            vec = l2_normalize(vec)
    except Exception as exc:
        logging.warning("[ERROR] embed %s: %s", img_path.name, exc)
        return "error"

    entry: dict[str, Any] = {
        "id": person_id,
        "nombre": nombre,
        "img": _rel_img_path(img_path),
        "rotacion": rotacion,
        "score": round(score, 4),
        "roll_deg": round(float(roll_deg), 1),
        "used_arcface_align": False,
        "used_roll_fix": False,
        "flip_avg": flip_avg,
    }
    vec_arr = np.asarray(vec, dtype=np.float32).reshape(EMBED_DIM)

    logging.info(
        "[OK] %s -> id=%s rotacion=%s (score=%.3f roll=%.1f)",
        img_path.name,
        person_id,
        rotacion,
        score,
        roll_deg,
    )
    return EnrollOk(vec=vec_arr, entry=entry)


def _save_gallery(build: GalleryBuild) -> None:
    gallery = np.stack(build.vectors, axis=0).astype(np.float32, copy=False)
    if gallery.shape != (len(build.entries), EMBED_DIM):
        raise RuntimeError(
            f"Shape inesperada: {gallery.shape} vs entries={len(build.entries)}"
        )

    meta = {
        "version": GALLERY_VERSION,
        "embed_dim": EMBED_DIM,
        "preprocess": GALLERY_PREPROCESS,
        "data_normalizada": GALLERY_DATA_NORMALIZADA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": build.entries,
        "detecciones": [_empty_deteccion() for _ in build.entries],
    }

    np.save(str(GALLERY_NPY), gallery)
    with GALLERY_META_JSON.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _record_result(stats: EnrollStats, result: EnrollOk | str, build: GalleryBuild) -> None:
    if isinstance(result, EnrollOk):
        build.vectors.append(result.vec)
        build.entries.append(result.entry)
        stats.ok += 1
        return

    if result == "skip_roll":
        stats.skip_roll += 1
    elif result == "skip_score":
        stats.skip_score += 1
    elif result == "skip_no_face":
        stats.skip_no_face += 1
    elif result == "skip_read":
        stats.skip_read += 1
    else:
        stats.errors += 1


def _enroll_images(
    images: list[Path],
    *,
    skip_err_ref: int,
    detector,
    embedder,
    flip_avg: bool,
) -> tuple[EnrollStats, GalleryBuild]:
    stats = EnrollStats(skip_err_ref=skip_err_ref)
    build = GalleryBuild()

    for img_path in images:
        if _is_or_reference(img_path):
            logging.warning(
                "[SKIP] referencia original (_or), no enrolar: %s",
                img_path.name,
            )
            stats.skip_or_ref += 1
            continue

        parsed = _parse_face_filename(img_path)
        if parsed is None:
            if _FILENAME_RE.match(img_path.name) is None:
                logging.warning(
                    "[SKIP] nombre invalido "
                    "(usar {id}_{nombre}[_zero|_der|_izq].ext): %s",
                    img_path.name,
                )
            stats.skip_invalid_name += 1
            continue

        person_id, nombre, rotacion = parsed
        result = _process_one_image(
            img_path,
            person_id=person_id,
            nombre=nombre,
            rotacion=rotacion,
            detector=detector,
            embedder=embedder,
            flip_avg=flip_avg,
        )
        _record_result(stats, result, build)

    return stats, build


def _log_enrollment_summary(stats: EnrollStats, *, candidates: int) -> None:
    total = candidates + stats.skip_err_ref
    logging.info(
        "Resumen: %d imagenes | OK=%d nombre=%d err_ref=%d or_ref=%d roll=%d score=%d "
        "sin_cara=%d lectura=%d error=%d",
        total,
        stats.ok,
        stats.skip_invalid_name,
        stats.skip_err_ref,
        stats.skip_or_ref,
        stats.skip_roll,
        stats.skip_score,
        stats.skip_no_face,
        stats.skip_read,
        stats.errors,
    )
    logging.info(
        "Galeria: %s shape=(%d, %d)",
        GALLERY_NPY.name,
        stats.ok,
        EMBED_DIM,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description=(
            "Enrolar gallery.npy + gallery_meta.json desde embeddings/faces_upd/ "
            "(solo crop)."
        )
    )
    p.add_argument(
        "--flip-avg",
        action="store_true",
        help="Promedio embedding + espejo horizontal (opcional).",
    )
    args = p.parse_args()

    s.validar_todo()
    if s.INFERENCE_BACKEND == "none":
        raise SystemExit(
            "INFERENCE_BACKEND=none: usar pc o rk3568 para enrolar embeddings."
        )

    if not FACES_DIR.is_dir():
        raise SystemExit(f"No existe carpeta de entrada: {FACES_DIR}")

    images, skip_err_ref = _list_images(FACES_DIR)
    if skip_err_ref:
        logging.info("Descartadas %d imagenes con prefijo err_", skip_err_ref)
    if not images:
        if skip_err_ref:
            raise SystemExit(
                f"Sin imagenes enrolables en {FACES_DIR} "
                f"({skip_err_ref} descartadas por prefijo err_)."
            )
        raise SystemExit(
            f"Sin imagenes en {FACES_DIR} (sufijos: {_IMAGE_SUFFIXES})"
        )

    _log_enrollment_config(flip_avg=args.flip_avg)

    detector = build_face_detector()
    embedder = build_embedder()
    if detector is None or embedder is None:
        raise SystemExit("No se pudo crear detector o embedder.")

    try:
        stats, build = _enroll_images(
            images,
            skip_err_ref=skip_err_ref,
            detector=detector,
            embedder=embedder,
            flip_avg=args.flip_avg,
        )
    finally:
        _release_inference(embedder, detector)

    if stats.ok == 0:
        raise SystemExit(
            "Ninguna persona enrolada. Revisar warnings (roll, score, nombre, etc.)."
        )

    _save_gallery(build)
    _log_enrollment_summary(stats, candidates=len(images))


if __name__ == "__main__":
    main()
