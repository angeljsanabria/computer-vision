import os
import sys
import logging

from configs.paths import resolve_repo_path

# Configuracion de logs para produccion
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 1. CONFIGURACIONES GENERALES
# 1.1 Captura
MODO = os.getenv("CONFIG_MODO", "USB").upper()     # RTSP, SNAP, USB
MAX_FPS = float(os.getenv("MAX_FPS", 60.0))
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", 15))
DISPLAY_IS_ENABLE = (
    os.getenv("DISPLAY_IS_ENABLE", "true").lower() == "true"
)
# RetinaFace a full rate (cada frame). Sin display conviene espaciarlo en
# FACE_RECOGNIZED (solo aporta el bbox del overlay). Default = DISPLAY_IS_ENABLE.
FACE_DETECT_FULLRATE = (
    os.getenv("FACE_DETECT_FULLRATE", str(DISPLAY_IS_ENABLE)).lower() == "true"
)
# Cuantas caras rankeadas procesar (1=mejor, 2=mejor+siguiente, ...).
FACE_PROCESS_TOP_N = int(os.getenv("FACE_PROCESS_TOP_N", 2))

# 1.2 Detalles de Captura
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1"))
CAP_FRAME_WIDTH = int(os.getenv("CAP_FRAME_WIDTH", 640))   #  High 2560    Medium 1080     Low 640
CAP_FRAME_HEIGHT = int(os.getenv("CAP_FRAME_HEIGHT", 480))  #  High 1920    Medium 720      Low 480
REINTENTO_SEG = float(os.getenv("REINTENTO_SEG", "10"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "10"))
LOG_CADA_N_FRAMES = int(os.getenv("LOG_CADA_N_FRAMES", "25"))

# 1.3 Procesamiento de imagen (RGA RK3568; legacy OpenCV por defecto)
USE_RGA = os.getenv("USE_RGA", "false").lower() == "true"

# 1.4 Identidad reconocida (FSM FACE_RECOGNIZED)
# Intervalo entre embeds en FACE_RECOGNIZED; cada MATCH renueva el timer de identidad.
# NO_MATCH con timer activo mantiene el ultimo MATCH; timer vencido -> FACE_PROCESSED.
FSM_RECOGNIZED_REFRESH_S = float(os.getenv("FSM_RECOGNIZED_REFRESH_S", "15"))

# 2. HARDWARE LOCAL (CAMARA USB)
USB_INDEX = int(os.getenv("USB_DEVICE_INDEX", 0))

# 3. CONFIGURACIONES Camara IP
_user = os.getenv("IP_CAM_USER", "angelcam")
_pass = os.getenv("IP_CAM_PASS", "angelCamara")
_host_ip = os.getenv("IP_CAM", "192.168.1.16")  # info dispositivo; info red

# 3.1 RTSP
_port = os.getenv("IP_CAM_RTSP_PORT", "554")   # info dispositivo; info avanzada
_route_rtsp_quality_low = os.getenv("IP_CAM_RTSP_ROUTE", "Preview_01_sub")
_route_rtsp_quality_high = os.getenv("IP_CAM_RTSP_ROUTE", "Preview_01_main")

IP_CAM_RTSP_URL = f"rtsp://{_user}:{_pass}@{_host_ip}:{_port}/{_route_rtsp_quality_low}"
IP_CAM_RTSP_URL_HIGH = f"rtsp://{_user}:{_pass}@{_host_ip}:{_port}/{_route_rtsp_quality_high}"

# 3.2 SNAP
_route_snap_quality_high = os.getenv("IP_CAM_ROUTE_SNAP_HIGH_RES", "width=2560&height=1920")
_route_snap_quality_low= os.getenv("IP_CAM_ROUTE_SNAP_LOW_RES", "width=640&height=480")

SNAP_HTTP_URL_RES_FULL = (
    f"http://{_host_ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={_user}&password={_pass}&{_route_snap_quality_high}"
)

SNAP_HTTP_URL_RES_LOW = (
    f"http://{_host_ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={_user}&password={_pass}&{_route_snap_quality_low}"
)

SNAP_HTTP_URL = SNAP_HTTP_URL_RES_LOW

# 4. DETECCION DE MOVIMIENTO (MOG2) + FSM
MOG2_PROCESS_WIDTH = int(os.getenv("MOG2_PROCESS_WIDTH", "320"))
MOG2_PROCESS_HEIGHT = int(os.getenv("MOG2_PROCESS_HEIGHT", "240"))
MOG2_HISTORY = int(os.getenv("MOG2_HISTORY", "20"))
MOG2_VAR_THRESHOLD = int(os.getenv("MOG2_VAR_THRESHOLD", "40"))
MOG2_MOVIMIENTO_PIXELES = int(os.getenv("MOG2_MOVIMIENTO_PIXELES", "750"))
MOG2_WARMUP_FRAMES = int(os.getenv("MOG2_WARMUP_FRAMES", "20"))
MOG2_WARMUP_LEARNING_RATE = float(os.getenv("MOG2_WARMUP_LEARNING_RATE", "0.5"))
MOG2_WARMUP_TIMEOUT_S = float(os.getenv("MOG2_WARMUP_TIMEOUT_S", "120"))
FSM_TIMEOUT_MOV_S = float(os.getenv("FSM_TIMEOUT_MOV_S", "10"))
FSM_TIMEOUT_FACE_S = float(os.getenv("FSM_TIMEOUT_FACE_S", "10"))

# 6. INFERENCIA (RetinaFace + MobileFaceNet)
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "pc").lower()  # "none", "pc", "rk3568"
RETINAFACE_MODEL_PC = os.getenv(
    "RETINAFACE_MODEL_PC",
    "models_onnx/RetinaFace_mobile320.onnx",
)
RETINAFACE_MODEL_RK3568 = os.getenv(
    "RETINAFACE_MODEL_RK3568",
    "models/RetinaFace_mobile320.rknn",
)
RETINAFACE_SCORE_DETECCION = float(os.getenv("RETINAFACE_SCORE_DETECCION", "0.5"))
RETINAFACE_SCORE_PRE_NMS = float(os.getenv("RETINAFACE_SCORE_PRE_NMS", "0.02"))

# 6.1 Preproceso cara para embedding
# Solo crop (defecto): ningun flag activo.
# FACE_ROT_ALIGNMENT_SIMPLE_ENABLE: hibrido crop / roll-fix si |roll| > FACE_ROLL_MAX_DEG.
# FACE_ALIGNMENT_ENABLE: siempre align ArcFace 5 pt (galeria .npy enrolada igual).
# Si ambos true, gana ArcFace (warning en validar_todo).
FACE_ALIGNMENT_ENABLE = (
    os.getenv("FACE_ALIGNMENT_ENABLE", "true").lower() == "true"
)
FACE_ROT_ALIGNMENT_SIMPLE_ENABLE = (
    os.getenv("FACE_ROT_ALIGNMENT_SIMPLE_ENABLE", "false").lower() == "true"
)
FACE_ROLL_MAX_DEG = float(os.getenv("FACE_ROLL_MAX_DEG", "10"))
FACE_CROP_MARGIN_FRAC = float(os.getenv("FACE_CROP_MARGIN_FRAC", "0.15"))

# 6.2 Embedding en FACE_PROCESSED (reglas de score y cooldown)
# Sin EMBED_MIN_SCORE en env: default RETINAFACE_SCORE_DETECCION (misma linea abajo).
# Debe cumplirse RETINAFACE_SCORE_DETECCION <= EMBED_MIN_SCORE (warning en validar_todo).
EMBED_MIN_SCORE = float(
    os.getenv("EMBED_MIN_SCORE", str(RETINAFACE_SCORE_DETECCION))
)
# Embed (FACE_PROCESSED/RECOGNIZED) y, sin FACE_DETECT_FULLRATE, RetinaFace en
# FACE_RECOGNIZED: como maximo cada EMBED_AND_FACEDETEC_COOLDOWN_S. 0 = cada tick con cara.
EMBED_AND_FACEDETEC_COOLDOWN_S = float(
    os.getenv("EMBED_AND_FACEDETEC_COOLDOWN_S", "1.0")
)

# 6.3 MobileFaceNet (rutas segun INFERENCE_BACKEND)
MOBILEFACENET_MODEL_PC = os.getenv(
    "MOBILEFACENET_MODEL_PC",
    "models_onnx/MobileFaceNet.onnx",
)
MOBILEFACENET_MODEL_RK3568 = os.getenv(
    "MOBILEFACENET_MODEL_RK3568",
    "models/MobileFaceNet.rknn",
)

# 6.4 Identidad (coseno vs galeria .npy; mismo criterio que RetinaFace_from_cam_with_id.py)
EMBED_SIM_MIN_MATCH = float(os.getenv("EMBED_SIM_MIN_MATCH", "0.55"))
EMBED_REF_GALLERY_DIR = os.getenv("EMBED_REF_GALLERY_DIR", "embeddings")

def retinaface_model_pc_path() -> str:
    """Ruta absoluta al ONNX RetinaFace (defecto: models_onnx/RetinaFace_mobile320.onnx)."""
    return str(resolve_repo_path(RETINAFACE_MODEL_PC))


def retinaface_model_rk3568_path() -> str:
    """Ruta absoluta al RKNN RetinaFace (defecto: models/RetinaFace_mobile320.rknn)."""
    return str(resolve_repo_path(RETINAFACE_MODEL_RK3568))


def mobilefacenet_model_pc_path() -> str:
    """Ruta absoluta al ONNX MobileFaceNet (defecto: models_onnx/MobileFaceNet.onnx)."""
    return str(resolve_repo_path(MOBILEFACENET_MODEL_PC))


def mobilefacenet_model_rk3568_path() -> str:
    """Ruta absoluta al RKNN MobileFaceNet (defecto: models/MobileFaceNet.rknn)."""
    return str(resolve_repo_path(MOBILEFACENET_MODEL_RK3568))


def embed_ref_gallery_dir_path() -> str:
    """Ruta absoluta a la galeria de referencias .npy (defecto: embeddings/)."""
    return str(resolve_repo_path(EMBED_REF_GALLERY_DIR))


def validar_todo():
    """Valida los parametros criticos en el arranque."""
    logging.info("=== VALIDANDO AJUSTES DE PRODUCCION ===")
    logging.info(
        f"Modo Activo: {MODO} | Velocidad Objetivo: {MAX_FPS} FPS | "
        f"Display: {DISPLAY_IS_ENABLE}"
    )

    if MODO not in ["RTSP", "SNAP", "USB"]:
        logging.critical(f"CONFIG ERROR: Modo '{MODO}' desconocido. Usar RTSP, SNAP o USB.")
        sys.exit(1)

    if MAX_FPS <= 0:
        logging.critical("CONFIG ERROR: MAX_FPS debe ser > 0.")
        sys.exit(1)

    if WARMUP_FRAMES < 1:
        logging.critical("CONFIG ERROR: WARMUP_FRAMES debe ser >= 1.")
        sys.exit(1)

    if MODO == "RTSP" and not _host_ip:
        logging.critical("CONFIG ERROR: Modo RTSP activo pero falta la IP (RTSP_HOST).")
        sys.exit(1)

    if MODO == "SNAP" and not SNAP_HTTP_URL:
        logging.critical("CONFIG ERROR: Modo SNAP sin URL configurada.")
        sys.exit(1)

    if MOG2_PROCESS_WIDTH < 1 or MOG2_PROCESS_HEIGHT < 1:
        logging.critical("CONFIG ERROR: MOG2_PROCESS_WIDTH/HEIGHT deben ser >= 1.")
        sys.exit(1)

    if MOG2_WARMUP_FRAMES < 1:
        logging.critical("CONFIG ERROR: MOG2_WARMUP_FRAMES debe ser >= 1.")
        sys.exit(1)

    if FSM_TIMEOUT_MOV_S <= 0 or FSM_TIMEOUT_FACE_S <= 0:
        logging.critical("CONFIG ERROR: FSM_TIMEOUT_MOV_S y FSM_TIMEOUT_FACE_S > 0.")
        sys.exit(1)
    if FSM_RECOGNIZED_REFRESH_S <= 0:
        logging.critical("CONFIG ERROR: FSM_RECOGNIZED_REFRESH_S debe ser > 0.")
        sys.exit(1)
    logging.info(
        "Identidad FSM: retencion MATCH %.1f s (FSM_RECOGNIZED_REFRESH_S)",
        FSM_RECOGNIZED_REFRESH_S,
    )

    if INFERENCE_BACKEND not in ("none", "pc", "rk3568"):
        logging.critical(
            "CONFIG ERROR: INFERENCE_BACKEND debe ser none, pc o rk3568."
        )
        sys.exit(1)

    if INFERENCE_BACKEND == "pc":
        pc_path = retinaface_model_pc_path()
        if not os.path.isfile(pc_path):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe RETINAFACE_MODEL_PC: "
                f"{pc_path}"
            )
            sys.exit(1)
        logging.info("RetinaFace PC: %s", pc_path)
        mfn_pc = mobilefacenet_model_pc_path()
        if not os.path.isfile(mfn_pc):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe "
                f"MOBILEFACENET_MODEL_PC: {mfn_pc}"
            )
            sys.exit(1)
        logging.info("MobileFaceNet PC: %s", mfn_pc)

    if INFERENCE_BACKEND == "rk3568":
        rk_path = retinaface_model_rk3568_path()
        if not os.path.isfile(rk_path):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe "
                f"RETINAFACE_MODEL_RK3568: {rk_path}"
            )
            sys.exit(1)
        logging.info("RetinaFace RK3568: %s", rk_path)
        mfn_rk = mobilefacenet_model_rk3568_path()
        if not os.path.isfile(mfn_rk):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe "
                f"MOBILEFACENET_MODEL_RK3568: {mfn_rk}"
            )
            sys.exit(1)
        logging.info("MobileFaceNet RK3568: %s", mfn_rk)

    if FACE_PROCESS_TOP_N < 1:
        logging.critical("CONFIG ERROR: FACE_PROCESS_TOP_N debe ser >= 1.")
        sys.exit(1)
    logging.info(
        "Caras a procesar: top %d (score >= RETINAFACE_SCORE_DETECCION=%.2f)",
        FACE_PROCESS_TOP_N,
        RETINAFACE_SCORE_DETECCION,
    )

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

    if FACE_ALIGNMENT_ENABLE and FACE_ROT_ALIGNMENT_SIMPLE_ENABLE:
        logging.warning(
            "Preproceso: FACE_ALIGNMENT_ENABLE y FACE_ROT_ALIGNMENT_SIMPLE_ENABLE "
            "activos; se usa solo align ArcFace (roll-fix ignorado)."
        )

    if FACE_ALIGNMENT_ENABLE:
        logging.info(
            "Preproceso cara: align ArcFace siempre (margen crop=%.2f). "
            "Enrolar refs con --preprocess arcface_align.",
            FACE_CROP_MARGIN_FRAC,
        )
    elif FACE_ROT_ALIGNMENT_SIMPLE_ENABLE:
        logging.info(
            "Preproceso cara: hibrido roll-fix (si |roll| > %.1f deg, margen=%.2f)",
            FACE_ROLL_MAX_DEG,
            FACE_CROP_MARGIN_FRAC,
        )
    else:
        logging.info(
            "Preproceso cara: solo crop bbox (margen=%.2f)",
            FACE_CROP_MARGIN_FRAC,
        )

    if EMBED_MIN_SCORE <= 0.0 or EMBED_MIN_SCORE > 1.0:
        logging.critical(
            "CONFIG ERROR: EMBED_MIN_SCORE debe estar en (0, 1] (got %.2f).",
            EMBED_MIN_SCORE,
        )
        sys.exit(1)

    if RETINAFACE_SCORE_DETECCION > EMBED_MIN_SCORE:
        logging.warning(
            "RETINAFACE_SCORE_DETECCION (%.2f) > EMBED_MIN_SCORE (%.2f): "
            "llegaran detecciones al pipeline de embed por debajo del score minimo "
            "de embed. Subir EMBED_MIN_SCORE o bajar RETINAFACE_SCORE_DETECCION.",
            RETINAFACE_SCORE_DETECCION,
            EMBED_MIN_SCORE,
        )

    if EMBED_AND_FACEDETEC_COOLDOWN_S < 0:
        logging.critical(
            "CONFIG ERROR: EMBED_AND_FACEDETEC_COOLDOWN_S debe ser >= 0 (got %.2f).",
            EMBED_AND_FACEDETEC_COOLDOWN_S,
        )
        sys.exit(1)

    if EMBED_AND_FACEDETEC_COOLDOWN_S == 0:
        logging.info(
            "Embed/deteccion: sin cooldown (cada tick con cara; util para metricas)"
        )
    else:
        logging.info(
            "Embed/deteccion: cooldown %.1f s (RetinaFace full rate=%s)",
            EMBED_AND_FACEDETEC_COOLDOWN_S,
            FACE_DETECT_FULLRATE,
        )

    if os.getenv("EMBED_MIN_SCORE") is not None:
        embed_min_src = "EMBED_MIN_SCORE (env)"
    else:
        embed_min_src = "RETINAFACE_SCORE_DETECCION (default)"
    logging.info(
        "Embed: score minimo %.2f (fuente %s)",
        EMBED_MIN_SCORE,
        embed_min_src,
    )

    if EMBED_SIM_MIN_MATCH < -1.0 or EMBED_SIM_MIN_MATCH > 1.0:
        logging.critical(
            "CONFIG ERROR: EMBED_SIM_MIN_MATCH debe estar en [-1, 1] (got %.2f).",
            EMBED_SIM_MIN_MATCH,
        )
        sys.exit(1)

    gallery_path = embed_ref_gallery_dir_path()
    if INFERENCE_BACKEND in ("pc", "rk3568"):
        if not os.path.isdir(gallery_path):
            logging.warning(
                "Galeria identidad: no existe directorio %s (EMBED_REF_GALLERY_DIR)",
                gallery_path,
            )
        else:
            if FACE_ALIGNMENT_ENABLE:
                npy_name, meta_name = "gallery_align.npy", "gallery_meta_align.json"
            else:
                npy_name, meta_name = "gallery.npy", "gallery_meta.json"
            npy_path = os.path.join(gallery_path, npy_name)
            meta_path = os.path.join(gallery_path, meta_name)
            has_matrix = os.path.isfile(npy_path) and os.path.isfile(meta_path)
            _gallery_npy_all = {"gallery.npy", "gallery_align.npy"}
            n_legacy = len(
                [
                    f
                    for f in os.listdir(gallery_path)
                    if f.lower().endswith(".npy") and f not in _gallery_npy_all
                ]
            )
            if has_matrix:
                logging.info(
                    "Galeria identidad: %s (%s + %s), sim_min_match=%.2f",
                    gallery_path,
                    npy_name,
                    meta_name,
                    EMBED_SIM_MIN_MATCH,
                )
            elif n_legacy > 0:
                logging.info(
                    "Galeria identidad: %s (%d .npy legacy), sim_min_match=%.2f",
                    gallery_path,
                    n_legacy,
                    EMBED_SIM_MIN_MATCH,
                )
            else:
                logging.warning(
                    "Galeria identidad: sin %s/%s ni .npy legacy en %s",
                    npy_name,
                    meta_name,
                    gallery_path,
                )
