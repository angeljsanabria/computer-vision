import os
import sys
import logging

# 1. CONFIGURACIONES GENERALES
# 1.1 Captura
MODO = os.getenv("CONFIG_MODO", "USB").upper()     # RTSP, SNAP, USB
MAX_FPS = float(os.getenv("MAX_FPS", 20.0))
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", 15))
DISPLAY_IS_ENABLE = (
    os.getenv("DISPLAY_IS_ENABLE", "true").lower() == "true"
)
DISPLAY_FORCE_FULL_SCREEN = (
    os.getenv("DISPLAY_FORCE_FULL_SCREEN", "false").lower() == "true"
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
LOG_MODE = os.getenv("LOG_MODE", "prod").lower()  # prod | dev

# 1.3 Procesamiento de imagen (RGA RK3568; legacy OpenCV por defecto)
USE_RGA = os.getenv("USE_RGA", "false").lower() == "true"

# 1.4 Identidad reconocida (FSM FACE_RECOGNIZED)
# Intervalo entre embeds en FACE_RECOGED; cada MATCH renueva el timer de identidad.
# NO_MATCH con timer activo mantiene el ultimo MATCH; timer vencido -> FACE_PROCESSED.
FSM_RECOGNIZED_REFRESH_S = float(os.getenv("FSM_RECOGNIZED_REFRESH_S", "15"))

# 1.5 API HTTP vision (modulo vision_http/; solo si ENABLE_ENDPOINT=true)
ENABLE_ENDPOINT = os.getenv("ENABLE_ENDPOINT", "true").lower() == "true"
HTTP_API_HOST = os.getenv("HTTP_API_HOST", "0.0.0.0")
HTTP_API_PORT = int(os.getenv("HTTP_API_PORT", "8008"))

# 1.6 Snapshot calidad imagen (headless; pisa img_snap.jpg cada N s en capture)
IMG_QUALITY_CHECK_ENABLE = (
    os.getenv("IMG_QUALITY_CHECK_ENABLE", "false").lower() == "true"
)
IMG_QUALITY_CHECK_INTERVAL_S = float(
    os.getenv("IMG_QUALITY_CHECK_INTERVAL_S", "30")
)
IMG_QUALITY_CHECK_DIR = os.getenv("IMG_QUALITY_CHECK_DIR", "data/")

# 2. HARDWARE LOCAL (CAMARA USB)
# Perfil de imagen OpenCV validado en camara Sony IMX179 /dev/video10 en RK3568.
# USE_DEFAULT: no modifica props de la camara.
# USE_CUSTOM: cap.set brillo/contraste/saturacion (valores abajo).
# Referencia V4L2 no usada por OpenCV aqui:
#   hue min=-180 max=180 default=0
#   gamma min=100 max=500 default=300
#   sharpness min=0 max=100 default=80
#   auto_exposure menu 0-3 default=3
#   focus_automatic_continuous default=1
USB_INDEX = int(os.getenv("USB_DEVICE_INDEX", 0))
USB_ROTATE_DEG = int(os.getenv("USB_ROTATE_DEG", "0"))  # 0, 90, 180, 270 (solo modo USB)
USB_CAMERA_IMAGE_MODE = os.getenv("USB_CAMERA_IMAGE_MODE", "USE_DEFAULT").upper()
USB_BRIGHTNESS = int(os.getenv("USB_BRIGHTNESS", "0"))  # SONY: min=-64 max=64 default=0
USB_CONTRAST = int(os.getenv("USB_CONTRAST", "51"))  # SONY: min=0 max=100 default=51
USB_SATURATION = int(os.getenv("USB_SATURATION", "64"))  # SONY: min=0 max=100 default=64

# 3. CONFIGURACIONES Camara IP (URLs en utils/ip_cam_urls.py)
IP_CAM_USER = os.getenv("IP_CAM_USER", "angelcam")
IP_CAM_PASS = os.getenv("IP_CAM_PASS", "angelCamara")
IP_CAM_HOST = os.getenv("IP_CAM", "172.16.243.10")

# 3.1 RTSP
IP_CAM_RTSP_PORT = os.getenv("IP_CAM_RTSP_PORT", "554")
IP_CAM_RTSP_STREAM_PATH_LOW = os.getenv("IP_CAM_RTSP_ROUTE_LOW", "Preview_01_sub")
IP_CAM_RTSP_STREAM_PATH_HIGH = os.getenv("IP_CAM_RTSP_ROUTE_HIGH", "Preview_01_main")

IP_CAM_RTSP_STREAM_PATH_SELECTED_RESOLUTION = IP_CAM_RTSP_STREAM_PATH_LOW

# 3.2 SNAP (query de resolucion; la URL se arma en main)
IP_CAM_SNAP_RES_QUERY_LOW = os.getenv("IP_CAM_ROUTE_SNAP_LOW_RES", "width=640&height=480")
IP_CAM_SNAP_RES_QUERY_HIGH = os.getenv("IP_CAM_ROUTE_SNAP_HIGH_RES", "width=2560&height=1920")

IP_CAM_SNAP_RES_QUERY_SELECTED_RESOLUTION = IP_CAM_SNAP_RES_QUERY_LOW

# 4. DETECCION DE MOVIMIENTO (MOG2) + FSM
ENABLE_MOV_DETECTION = (
    os.getenv("ENABLE_MOV_DETECTION", "false").lower() == "true"
)
    
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
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "PC").lower()  # "none", "pc", "rk3568"
RETINAFACE_MODEL_PC = os.getenv(
    "RETINAFACE_MODEL_PC",
    "models_onnx/RetinaFace_mobile320.onnx",
)
RETINAFACE_MODEL_RK3568 = os.getenv(
    "RETINAFACE_MODEL_RK3568",
    "models/RetinaFace_mobile320.rknn",
)
RETINAFACE_SCORE_DETECCION = float(os.getenv("RETINAFACE_SCORE_DETECCION", "0.8"))
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
    os.getenv("EMBED_AND_FACEDETEC_COOLDOWN_S", "2.0")
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
EMBED_REF_GALLERY_DIR = os.getenv("EMBED_REF_GALLERY_DIR", "data/")

# 7. TRACKING VISUAL (ByteTrack sobre detecciones RetinaFace ya filtradas)
# Solo overlay/UI: no altera embed, matcher ni FSM. dets se lee, nunca se muta.
ENABLE_FACE_TRACKING = (
    os.getenv("ENABLE_FACE_TRACKING", "false").lower() == "true"
)
# Score minimo para asociacion de alta confianza / activar tracks nuevos.
# Por defecto igual a RETINAFACE_SCORE_DETECCION (RetinaFace ya filtra ahi).
BYTETRACK_TRACK_THRESH = float(
    os.getenv("BYTETRACK_TRACK_THRESH", str(RETINAFACE_SCORE_DETECCION))
)
# Umbral de costo IoU en la asociacion deteccion-track (mas alto = mas estricto).
BYTETRACK_MATCH_THRESH = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.8"))
# Ventana (a 30 FPS) de frames que un track puede estar perdido antes de expirar.
BYTETRACK_TRACK_BUFFER = int(os.getenv("BYTETRACK_TRACK_BUFFER", "30"))
# FPS real del pipeline para escalar el buffer temporal (defecto = MAX_FPS).
BYTETRACK_FRAME_RATE = float(os.getenv("BYTETRACK_FRAME_RATE", str(MAX_FPS)))


_LOG_LEVEL_BY_MODE = {
    "prod": logging.INFO,
    "dev": logging.DEBUG,
}


def _resolve_log_level() -> int:
    return _LOG_LEVEL_BY_MODE.get(LOG_MODE, logging.INFO)


def configure_logging() -> None:
    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def validar_todo():
    """Valida los parametros criticos en el arranque."""
    logging.info("=== VALIDANDO AJUSTES DE PRODUCCION ===")
    logging.info(
        f"Modo Activo: {MODO} | Velocidad Objetivo: {MAX_FPS} FPS | "
        f"Display: {DISPLAY_IS_ENABLE}"
    )
    if DISPLAY_IS_ENABLE and DISPLAY_FORCE_FULL_SCREEN:
        logging.info(f"Force Full Screen: {DISPLAY_FORCE_FULL_SCREEN}")

    if MODO not in ["RTSP", "SNAP", "USB"]:
        logging.critical(f"CONFIG ERROR: Modo '{MODO}' desconocido. Usar RTSP, SNAP o USB.")
        sys.exit(1)

    if MAX_FPS <= 0:
        logging.critical("CONFIG ERROR: MAX_FPS debe ser > 0.")
        sys.exit(1)

    if WARMUP_FRAMES < 1:
        logging.critical("CONFIG ERROR: WARMUP_FRAMES debe ser >= 1.")
        sys.exit(1)

    if MODO in ("RTSP", "SNAP") and not IP_CAM_HOST:
        logging.critical("CONFIG ERROR: Modo %s activo pero falta IP_CAM.", MODO)
        sys.exit(1)

    if USB_ROTATE_DEG not in (0, 90, 180, 270):
        logging.critical(
            "CONFIG ERROR: USB_ROTATE_DEG debe ser 0, 90, 180 o 270 (got %d).",
            USB_ROTATE_DEG,
        )
        sys.exit(1)
    if MODO == "USB" and USB_ROTATE_DEG != 0:
        logging.info("USB: rotacion software %d deg", USB_ROTATE_DEG)

    if MODO == "USB":
        if USB_CAMERA_IMAGE_MODE not in ("USE_DEFAULT", "USE_CUSTOM"):
            logging.critical(
                "CONFIG ERROR: USB_CAMERA_IMAGE_MODE debe ser USE_DEFAULT o "
                "USE_CUSTOM (got %r).",
                USB_CAMERA_IMAGE_MODE,
            )
            sys.exit(1)
        if USB_CAMERA_IMAGE_MODE == "USE_CUSTOM":
            logging.info("USB: ajuste imagen OpenCV USE_CUSTOM")

    if not ENABLE_MOV_DETECTION:
        logging.info(
            "MOG2 desactivado: FSM usa FACE_LOOKING (RetinaFace sin gate de movimiento)"
        )
        
    if ENABLE_ENDPOINT:
        if HTTP_API_PORT < 1 or HTTP_API_PORT > 65535:
            logging.critical(
                "CONFIG ERROR: HTTP_API_PORT debe estar en [1, 65535] (got %d).",
                HTTP_API_PORT,
            )
            sys.exit(1)
        logging.info(
            "API vision: ENABLE_ENDPOINT en %s:%d (/api/v1/vision-status)",
            HTTP_API_HOST,
            HTTP_API_PORT,
        )

    if IMG_QUALITY_CHECK_ENABLE:
        if IMG_QUALITY_CHECK_INTERVAL_S <= 0:
            logging.critical(
                "CONFIG ERROR: IMG_QUALITY_CHECK_INTERVAL_S debe ser > 0 (got %.3f).",
                IMG_QUALITY_CHECK_INTERVAL_S,
            )
            sys.exit(1)
        logging.info(
            "img_quality_check: cada %.1f s -> %s/img_snap.jpg",
            IMG_QUALITY_CHECK_INTERVAL_S,
            IMG_QUALITY_CHECK_DIR,
        )

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
        pc_path = RETINAFACE_MODEL_PC
        if not os.path.isfile(pc_path):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe RETINAFACE_MODEL_PC: "
                f"{pc_path}"
            )
            sys.exit(1)
        logging.info("RetinaFace PC: %s", pc_path)
        mfn_pc = MOBILEFACENET_MODEL_PC
        if not os.path.isfile(mfn_pc):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=pc pero no existe "
                f"MOBILEFACENET_MODEL_PC: {mfn_pc}"
            )
            sys.exit(1)
        logging.info("MobileFaceNet PC: %s", mfn_pc)

    if INFERENCE_BACKEND == "rk3568":
        rk_path = RETINAFACE_MODEL_RK3568
        if not os.path.isfile(rk_path):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe "
                f"RETINAFACE_MODEL_RK3568: {rk_path}"
            )
            sys.exit(1)
        logging.info("RetinaFace RK3568: %s", rk_path)
        mfn_rk = MOBILEFACENET_MODEL_RK3568
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

    gallery_path = EMBED_REF_GALLERY_DIR
    if INFERENCE_BACKEND in ("pc", "rk3568"):
        if not os.path.isdir(gallery_path):
            logging.critical(
                "CONFIG ERROR: Galeria identidad: no existe directorio %s "
                "(EMBED_REF_GALLERY_DIR). Sin galeria no hay reconocimiento posible.",
                gallery_path,
            )
            sys.exit(1)
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
                logging.critical(
                    "CONFIG ERROR: Galeria identidad: sin %s/%s ni .npy legacy en %s. "
                    "Sin galeria no hay reconocimiento posible.",
                    npy_name,
                    meta_name,
                    gallery_path,
                )
                sys.exit(1)

    if ENABLE_FACE_TRACKING:
        if BYTETRACK_TRACK_THRESH <= 0.0 or BYTETRACK_TRACK_THRESH > 1.0:
            logging.critical(
                "CONFIG ERROR: BYTETRACK_TRACK_THRESH debe estar en (0, 1] (got %.2f).",
                BYTETRACK_TRACK_THRESH,
            )
            sys.exit(1)
        if BYTETRACK_MATCH_THRESH <= 0.0 or BYTETRACK_MATCH_THRESH > 1.0:
            logging.critical(
                "CONFIG ERROR: BYTETRACK_MATCH_THRESH debe estar en (0, 1] (got %.2f).",
                BYTETRACK_MATCH_THRESH,
            )
            sys.exit(1)
        if BYTETRACK_TRACK_BUFFER < 1:
            logging.critical(
                "CONFIG ERROR: BYTETRACK_TRACK_BUFFER debe ser >= 1 (got %d).",
                BYTETRACK_TRACK_BUFFER,
            )
            sys.exit(1)
        if BYTETRACK_FRAME_RATE <= 0.0:
            logging.critical(
                "CONFIG ERROR: BYTETRACK_FRAME_RATE debe ser > 0 (got %.2f).",
                BYTETRACK_FRAME_RATE,
            )
            sys.exit(1)
        logging.info(
            "ByteTrack: activo (track_thresh=%.2f, match_thresh=%.2f, "
            "buffer=%d frames @ %.1f fps)",
            BYTETRACK_TRACK_THRESH,
            BYTETRACK_MATCH_THRESH,
            BYTETRACK_TRACK_BUFFER,
            BYTETRACK_FRAME_RATE,
        )
    else:
        logging.info("ByteTrack: desactivado (ENABLE_FACE_TRACKING=false)")


configure_logging()
