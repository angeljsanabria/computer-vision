"""
Settings reducido para el enrolamiento OFFLINE de galeria facial.

Vive junto a los scripts que lo usan (prepare_faces_refs.py,
face_embeddings_npy_from_images_folder.py, face_embeddings_align_npy_from_images_folder.py)
para poder importarlo directo (``import settings_embedder as s``). NO depende de
configs/settings.py: este modulo es la unica fuente de ajustes para enrolar.

Al importarse localiza la carpeta ``src/`` (la que contiene ``inference/``) y la
agrega a ``sys.path`` para poder ``import inference`` (RetinaFace + MobileFaceNet).
Las rutas de modelos por defecto se resuelven contra ``src/`` (los .onnx viven en
``src/models_onnx/``), de modo que los scripts funcionan desde cualquier CWD.

Subconjunto de configs/settings.py: RetinaFace + MobileFaceNet + preproceso de
cara. No incluye camara, MOG2, FSM, HTTP ni ByteTrack: eso es exclusivo del
runtime en vivo (main.py).
"""
import os
import sys
import logging
from pathlib import Path


def _find_src_dir() -> Path:
    """
    Localiza la carpeta ``src/`` que contiene el paquete ``inference/``.

    Busca desde este archivo hacia arriba: en cada ancestro prueba ``<d>/src``
    y el propio ``<d>`` (por si el script se moviera dentro de src/).
    """
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        candidate = d / "src"
        if (candidate / "inference" / "__init__.py").is_file():
            return candidate
        if (d / "inference" / "__init__.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro la carpeta src/ (con inference/) desde " + str(here)
    )


SRC_DIR = _find_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _resolve_model_path(path: str) -> str:
    """Ruta absoluta: si es relativa, se resuelve contra src/ (no contra el CWD)."""
    p = Path(path)
    if not p.is_absolute():
        p = SRC_DIR / p
    return str(p)


INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "PC").lower()  # "none", "pc", "rk3568"

RETINAFACE_MODEL_PC = _resolve_model_path(os.getenv(
    "RETINAFACE_MODEL_PC",
    "models_onnx/RetinaFace_mobile320.onnx",
))
RETINAFACE_MODEL_RK3568 = _resolve_model_path(os.getenv(
    "RETINAFACE_MODEL_RK3568",
    "models/RetinaFace_mobile320.rknn",
))
RETINAFACE_SCORE_DETECCION = float(os.getenv("RETINAFACE_SCORE_DETECCION", "0.8"))
RETINAFACE_SCORE_PRE_NMS = float(os.getenv("RETINAFACE_SCORE_PRE_NMS", "0.02"))

MOBILEFACENET_MODEL_PC = _resolve_model_path(os.getenv(
    "MOBILEFACENET_MODEL_PC",
    "models_onnx/MobileFaceNet.onnx",
))
MOBILEFACENET_MODEL_RK3568 = _resolve_model_path(os.getenv(
    "MOBILEFACENET_MODEL_RK3568",
    "models/MobileFaceNet.rknn",
))

# Preproceso cara (identico criterio que settings.py Seccion 6.1, sin el resto del pipeline)
FACE_ROLL_MAX_DEG = float(os.getenv("FACE_ROLL_MAX_DEG", "10"))
FACE_CROP_MARGIN_FRAC = float(os.getenv("FACE_CROP_MARGIN_FRAC", "0.15"))
USE_RGA = os.getenv("USE_RGA", "false").lower() == "true"

LOG_MODE = os.getenv("LOG_MODE", "prod").lower()  # prod | dev

_LOG_LEVEL_BY_MODE = {
    "prod": logging.INFO,
    "dev": logging.DEBUG,
}


def resolve_model_paths() -> tuple[str, str]:
    """Rutas (retinaface, mobilefacenet) segun INFERENCE_BACKEND. ("", "") si backend invalido/none."""
    if INFERENCE_BACKEND == "pc":
        return RETINAFACE_MODEL_PC, MOBILEFACENET_MODEL_PC
    if INFERENCE_BACKEND == "rk3568":
        return RETINAFACE_MODEL_RK3568, MOBILEFACENET_MODEL_RK3568
    return "", ""


def _resolve_log_level() -> int:
    return _LOG_LEVEL_BY_MODE.get(LOG_MODE, logging.INFO)


def configure_logging() -> None:
    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def validar_todo() -> None:
    """Valida los parametros criticos para enrolar (subconjunto de settings.validar_todo)."""
    logging.info("=== VALIDANDO AJUSTES DE ENROLAMIENTO (settings_embedder) ===")

    if INFERENCE_BACKEND not in ("none", "pc", "rk3568"):
        logging.critical(
            "CONFIG ERROR: INFERENCE_BACKEND debe ser none, pc o rk3568 (got %r).",
            INFERENCE_BACKEND,
        )
        sys.exit(1)

    if INFERENCE_BACKEND == "pc":
        if not os.path.isfile(RETINAFACE_MODEL_PC):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe RETINAFACE_MODEL_PC: %s",
                RETINAFACE_MODEL_PC,
            )
            sys.exit(1)
        logging.info("RetinaFace PC: %s", RETINAFACE_MODEL_PC)
        if not os.path.isfile(MOBILEFACENET_MODEL_PC):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe MOBILEFACENET_MODEL_PC: %s",
                MOBILEFACENET_MODEL_PC,
            )
            sys.exit(1)
        logging.info("MobileFaceNet PC: %s", MOBILEFACENET_MODEL_PC)

    if INFERENCE_BACKEND == "rk3568":
        if not os.path.isfile(RETINAFACE_MODEL_RK3568):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe RETINAFACE_MODEL_RK3568: %s",
                RETINAFACE_MODEL_RK3568,
            )
            sys.exit(1)
        logging.info("RetinaFace RK3568: %s", RETINAFACE_MODEL_RK3568)
        if not os.path.isfile(MOBILEFACENET_MODEL_RK3568):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe MOBILEFACENET_MODEL_RK3568: %s",
                MOBILEFACENET_MODEL_RK3568,
            )
            sys.exit(1)
        logging.info("MobileFaceNet RK3568: %s", MOBILEFACENET_MODEL_RK3568)

    if FACE_ROLL_MAX_DEG < 0 or FACE_ROLL_MAX_DEG > 45:
        logging.critical(
            "CONFIG ERROR: FACE_ROLL_MAX_DEG debe estar entre 0 y 45 (got %.1f).",
            FACE_ROLL_MAX_DEG,
        )
        sys.exit(1)

    if FACE_CROP_MARGIN_FRAC < 0 or FACE_CROP_MARGIN_FRAC >= 1.0:
        logging.critical(
            "CONFIG ERROR: FACE_CROP_MARGIN_FRAC debe estar en [0, 1) (got %.2f).",
            FACE_CROP_MARGIN_FRAC,
        )
        sys.exit(1)

    logging.info(
        "Backend: %s | crop_margin=%.2f | roll_max=%.1f deg",
        INFERENCE_BACKEND,
        FACE_CROP_MARGIN_FRAC,
        FACE_ROLL_MAX_DEG,
    )


configure_logging()
