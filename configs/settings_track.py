import os
import sys
import logging

# 1. CONFIGURACIONES GENERALES
# 1.0 Plataforma
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "PC").lower()  # "none", "pc", "rk3568"

# 1.1 Captura
MODO = os.getenv("CONFIG_MODO", "USB").upper()     # RTSP, RTMP, SNAP, USB
MAX_FPS = float(os.getenv("MAX_FPS", 20.0))
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", 15))
DISPLAY_IS_ENABLE = (
    os.getenv("DISPLAY_IS_ENABLE", "true").lower() == "true"
)
DISPLAY_FORCE_FULL_SCREEN = (
    os.getenv("DISPLAY_FORCE_FULL_SCREEN", "true").lower() == "true"
)
# Tamano ventana OpenCV (resizeWindow + moveWindow). 0 desactiva el resize.
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "1920"))         # 0
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "1080"))       # 0

# Banner superior (demo). Vacio = desactivado.
# try_resolve_from_path (desde main): busca {stem}_{DISPLAY_WIDTH}.png|.jpg
# en la misma carpeta; fallback al archivo de esta ruta (ej. baner_test.jpg).
DISPLAY_BANNER_PATH = os.getenv("DISPLAY_BANNER_PATH", "../data/banner.jpg")

# RetinaFace a full rate (cada frame). Sin display conviene espaciarlo en
# FACE_RECOGNIZED (solo aporta el bbox del overlay). Default = DISPLAY_IS_ENABLE.
FACE_DETECT_FULLRATE = (
    os.getenv("FACE_DETECT_FULLRATE", str(DISPLAY_IS_ENABLE)).lower() == "true"
)
# Cuantas caras mostrar (bbox/track) entre las que pasan RETINAFACE_SCORE_DETECCION.
# Ej.: 20 detectadas, 10 pasan umbral, N=5 -> las 5 mejores en pantalla.
FACE_PROCESS_TOP_N = int(os.getenv("FACE_PROCESS_TOP_N", 10))
# De esas N rankeadas, cuantas reciben embed + reconocimiento (mejores primero).
FACE_EMBED_TOP_N = int(os.getenv("FACE_EMBED_TOP_N", 2))
# FaceMesh UX: landmarks solo en tracks sin MATCH (desconocidos). Requiere display.
ENABLE_FACEMESH = os.getenv("ENABLE_FACEMESH", "true").lower() == "true"
# Maximo de desconocidos con mesh por frame (presupuesto CPU; tipico 2-3).
FACE_MESH_TOP_N = int(os.getenv("FACE_MESH_TOP_N", 2))
# Inferir cada N frames; el resto reusa hold (throttle). 1 = cada frame.
# Hold tambien cubre frames sin det RetinaFace (anti-parpadeo).
FACE_MESH_EVERY_N_FRAMES = int(os.getenv("FACE_MESH_EVERY_N_FRAMES", "2"))

# 1.2 Detalles de Captura
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1"))
CAP_FRAME_WIDTH = int(os.getenv("CAP_FRAME_WIDTH", 1920))   #  High 2560    Medium 1080     Low 640
CAP_FRAME_HEIGHT = int(os.getenv("CAP_FRAME_HEIGHT", 1080))  #  High 1920    Medium 720      Low 480
REINTENTO_SEG = float(os.getenv("REINTENTO_SEG", "10"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "10"))
LOG_CADA_N_FRAMES = int(os.getenv("LOG_CADA_N_FRAMES", "25"))
LOG_MODE = os.getenv("LOG_MODE", "prod").lower()  # prod | dev

# 1.3 Procesamiento de imagen (RGA RK3568; solo efectivo con INFERENCE_BACKEND=rk3568)
USE_RGA = os.getenv("USE_RGA", "false").lower() == "true"

# 1.4 Identidad reconocida (FSM FACE_RECOGNIZED)
# Intervalo entre embeds en FACE_RECOGED; cada MATCH renueva el timer de identidad.
# NO_MATCH con timer activo mantiene el ultimo MATCH; timer vencido -> FACE_PROCESSED.
FSM_RECOGNIZED_REFRESH_S = float(os.getenv("FSM_RECOGNIZED_REFRESH_S", "15"))

# 1.5 API HTTP vision (modulo vision_http/; solo si ENABLE_ENDPOINT=true)
ENABLE_ENDPOINT = os.getenv("ENABLE_ENDPOINT", "false").lower() == "true"
HTTP_API_HOST = os.getenv("HTTP_API_HOST", "0.0.0.0")
HTTP_API_PORT = int(os.getenv("HTTP_API_PORT", "8008"))

# 1.6 Snapshot calidad imagen (headless; pisa img_snap.jpg cada N s en capture)
IMG_QUALITY_CHECK_ENABLE = (
    os.getenv("IMG_QUALITY_CHECK_ENABLE", "false").lower() == "true"
)
IMG_QUALITY_CHECK_INTERVAL_S = float(
    os.getenv("IMG_QUALITY_CHECK_INTERVAL_S", "30")
)
IMG_QUALITY_CHECK_DIR = os.getenv("IMG_QUALITY_CHECK_DIR", "../data")

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
IP_CAM_USER = os.getenv("IP_CAM_USER", "CtkSom1")
IP_CAM_PASS = os.getenv("IP_CAM_PASS", "CtkSom1pass")
IP_CAM_HOST = os.getenv("IP_CAM", "172.16.243.10")

# 3.1 RTSP
IP_CAM_RTSP_PORT = os.getenv("IP_CAM_RTSP_PORT", "554")
IP_CAM_RTSP_STREAM_PATH_LOW = os.getenv("IP_CAM_RTSP_ROUTE_LOW", "Preview_01_sub")
IP_CAM_RTSP_STREAM_PATH_HIGH = os.getenv("IP_CAM_RTSP_ROUTE_HIGH", "Preview_01_main")

IP_CAM_RTSP_STREAM_PATH_SELECTED_RESOLUTION = IP_CAM_RTSP_STREAM_PATH_LOW

# 3.2 RTMP (Reolink standalone; CONFIG_MODO=RTMP)
# IP_CAM_USE_RTMP_BALANCED quedo deprecado: usar CONFIG_MODO=RTMP.
IP_CAM_RTMP_PORT = os.getenv("IP_CAM_RTMP_PORT", "1935")
IP_CAM_RTMP_STREAM_MAIN = "MAIN"
IP_CAM_RTMP_STREAM_EXT = "EXT"
IP_CAM_RTMP_STREAM_SUB = "SUB"
IP_CAM_RTMP_STREAM_SELECTED = os.getenv(
    "IP_CAM_RTMP_STREAM", IP_CAM_RTMP_STREAM_EXT
).upper()

# 3.3 SNAP (query de resolucion; la URL se arma en main)
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

RETINAFACE_MODEL_PC = os.getenv(
    "RETINAFACE_MODEL_PC",
    "models_onnx/RetinaFace_mobile320.onnx",
)
RETINAFACE_MODEL_RK3568 = os.getenv(
    "RETINAFACE_MODEL_RK3568",
    "models/RetinaFace_mobile320.rknn",
)
RETINAFACE_SCORE_DETECCION = float(os.getenv("RETINAFACE_SCORE_DETECCION", "0.4"))      # Modificado para el tracking
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
    os.getenv("EMBED_MIN_SCORE", "0.75")
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

# 6.3b FaceMesh 468 landmarks (UX desconocidos; rutas segun INFERENCE_BACKEND)
FACEMESH_MODEL_PC = os.getenv(
    "FACEMESH_MODEL_PC",
    "models_onnx/face_mesh_192x192.onnx",
)
FACEMESH_MODEL_RK3568 = os.getenv(
    "FACEMESH_MODEL_RK3568",
    "models/face_mesh_192x192.rknn",
)

# 6.4 Identidad (coseno vs galeria .npy; mismo criterio que RetinaFace_from_cam_with_id.py)
EMBED_SIM_MIN_MATCH = float(os.getenv("EMBED_SIM_MIN_MATCH", "0.55"))
EMBED_REF_GALLERY_DIR = os.getenv("EMBED_REF_GALLERY_DIR", "../data")

# 7. TRACKING VISUAL (ByteTrack sobre detecciones RetinaFace ya filtradas)
# Solo overlay/UI: no altera embed, matcher ni FSM. dets se lee, nunca se muta.
ENABLE_FACE_TRACKING = (
    os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true"
)
# Score minimo para asociacion de alta confianza / activar tracks nuevos.
# Default fijo (NO sigue a RETINAFACE_SCORE_DETECCION): si se baja RETINAFACE_SCORE_DETECCION
# para sostener continuidad de tracking con detecciones debiles, este umbral debe mantenerse
# alto para que ByteTrack solo cree tracks nuevos con deteccion confiable (ver segundo nivel
# de asociacion "baja confianza" en bytetrack/byte_tracker.py, piso interno 0.1).
BYTETRACK_TRACK_THRESH = float(os.getenv("BYTETRACK_TRACK_THRESH", "0.75"))  # entran al pull de confianza de tracking
# Umbral de costo IoU en la asociacion deteccion-track (mas alto = mas estricto).
BYTETRACK_MATCH_THRESH = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.7"))  # umbral de asociacion de deteccion-track
# Mas bajo tolera mas movimiento ; Mas alto es mas exigente (tolera menos movimiento)
# Ventana (a 30 FPS) de frames que un track puede estar perdido antes de expirar.
BYTETRACK_TRACK_BUFFER = int(os.getenv("BYTETRACK_TRACK_BUFFER", "20"))
# FPS real del pipeline para escalar el buffer temporal (defecto = MAX_FPS).
BYTETRACK_FRAME_RATE = float(os.getenv("BYTETRACK_FRAME_RATE", str(MAX_FPS)))

# 7.1 Nozzle YOLOv8 (deteccion + ByteTrack paralelo al pipeline facial; solo overlay/log)
ENABLE_NOZZLE = os.getenv("ENABLE_NOZZLE", "true").lower() == "true"
NOZZLE_MODEL_PC = os.getenv(
    "NOZZLE_MODEL_PC",
    "models_onnx/yolov8n_nozzle_v2.onnx",
)
NOZZLE_MODEL_RK3568 = os.getenv(
    "NOZZLE_MODEL_RK3568",
    "models/yolov8n_nozzle_v2.rknn",
)
NOZZLE_SCORE_DETECCION = float(os.getenv("NOZZLE_SCORE_DETECCION", "0.30"))
NOZZLE_NMS_IOU = float(os.getenv("NOZZLE_NMS_IOU", "0.45"))
NOZZLE_PROCESS_TOP_N = int(os.getenv("NOZZLE_PROCESS_TOP_N", "10"))
NOZZLE_EVERY_N_FRAMES = int(os.getenv("NOZZLE_EVERY_N_FRAMES", "1"))
NOZZLE_BYTETRACK_TRACK_THRESH = float(
    os.getenv("NOZZLE_BYTETRACK_TRACK_THRESH", "0.65")
)
NOZZLE_BYTETRACK_MATCH_THRESH = float(
    os.getenv("NOZZLE_BYTETRACK_MATCH_THRESH", "0.70")
)
NOZZLE_BYTETRACK_TRACK_BUFFER = int(os.getenv("NOZZLE_BYTETRACK_TRACK_BUFFER", "20"))
NOZZLE_BYTETRACK_FRAME_RATE = float(
    os.getenv("NOZZLE_BYTETRACK_FRAME_RATE", str(MAX_FPS))
)


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
    """Valida los parametros criticos en el arranque (perfil TRACK)."""
    logging.info("=== VALIDANDO AJUSTES (PERFIL TRACK) ===")
    logging.info(
        f"Modo Activo: {MODO} | Velocidad Objetivo: {MAX_FPS} FPS | "
        f"Display: {DISPLAY_IS_ENABLE}"
    )
    if DISPLAY_IS_ENABLE and DISPLAY_FORCE_FULL_SCREEN:
        logging.info(f"Force Full Screen: {DISPLAY_FORCE_FULL_SCREEN}")
    if DISPLAY_IS_ENABLE and DISPLAY_WIDTH > 0 and DISPLAY_HEIGHT > 0:
        logging.info(
            "Display ventana: %dx%d (letterbox negro en show)",
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )
    if DISPLAY_IS_ENABLE and (DISPLAY_WIDTH != 0 or DISPLAY_HEIGHT != 0) and not (
        DISPLAY_WIDTH > 0 and DISPLAY_HEIGHT > 0
    ):
        logging.critical(
            "CONFIG ERROR: DISPLAY_WIDTH y DISPLAY_HEIGHT deben ser ambos > 0 o ambos 0."
        )
        sys.exit(1)

    if MODO not in ["RTSP", "RTMP", "SNAP", "USB"]:
        logging.critical(
            "CONFIG ERROR: Modo '%s' desconocido. Usar RTSP, RTMP, SNAP o USB.",
            MODO,
        )
        sys.exit(1)

    if MAX_FPS <= 0:
        logging.critical("CONFIG ERROR: MAX_FPS debe ser > 0.")
        sys.exit(1)

    if WARMUP_FRAMES < 1:
        logging.critical("CONFIG ERROR: WARMUP_FRAMES debe ser >= 1.")
        sys.exit(1)

    if MODO in ("RTSP", "RTMP", "SNAP") and not IP_CAM_HOST:
        logging.critical("CONFIG ERROR: Modo %s activo pero falta IP_CAM.", MODO)
        sys.exit(1)

    if os.getenv("IP_CAM_USE_RTMP_BALANCED", "").lower() == "true":
        logging.warning(
            "IP_CAM_USE_RTMP_BALANCED esta deprecado; usar CONFIG_MODO=RTMP "
            "(perfil: IP_CAM_RTMP_STREAM=MAIN|EXT|SUB)."
        )

    if MODO == "RTMP":
        if IP_CAM_RTMP_STREAM_SELECTED not in (
            IP_CAM_RTMP_STREAM_MAIN,
            IP_CAM_RTMP_STREAM_EXT,
            IP_CAM_RTMP_STREAM_SUB,
        ):
            logging.critical(
                "CONFIG ERROR: IP_CAM_RTMP_STREAM debe ser MAIN, EXT o SUB (got %r).",
                IP_CAM_RTMP_STREAM_SELECTED,
            )
            sys.exit(1)
        logging.info(
            "Camara IP: RTMP perfil %s (puerto %s)",
            IP_CAM_RTMP_STREAM_SELECTED,
            IP_CAM_RTMP_PORT,
        )

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

    if USE_RGA and INFERENCE_BACKEND != "rk3568":
        logging.critical(
            "CONFIG ERROR: USE_RGA=true requiere INFERENCE_BACKEND=rk3568 (actual: %s).",
            INFERENCE_BACKEND,
        )
        sys.exit(1)
    if USE_RGA:
        logging.info(
            "RGA activo (rk3568): resize/letterbox/cvtColor via my_rga "
            "(fallback OpenCV si falta wheel)"
        )

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
    if FACE_EMBED_TOP_N < 1:
        logging.critical("CONFIG ERROR: FACE_EMBED_TOP_N debe ser >= 1.")
        sys.exit(1)
    if FACE_EMBED_TOP_N > FACE_PROCESS_TOP_N:
        logging.warning(
            "FACE_EMBED_TOP_N (%d) > FACE_PROCESS_TOP_N (%d): "
            "se limitara embed a %d.",
            FACE_EMBED_TOP_N,
            FACE_PROCESS_TOP_N,
            FACE_PROCESS_TOP_N,
        )
    logging.info(
        "Caras bbox/track: top %d (score >= RETINAFACE_SCORE_DETECCION=%.2f); "
        "embed: top %d de esas",
        FACE_PROCESS_TOP_N,
        RETINAFACE_SCORE_DETECCION,
        min(FACE_EMBED_TOP_N, FACE_PROCESS_TOP_N),
    )

    if ENABLE_FACEMESH:
        if FACE_MESH_TOP_N < 1:
            logging.critical("CONFIG ERROR: FACE_MESH_TOP_N debe ser >= 1.")
            sys.exit(1)
        if FACE_MESH_EVERY_N_FRAMES < 1:
            logging.critical(
                "CONFIG ERROR: FACE_MESH_EVERY_N_FRAMES debe ser >= 1 (got %d).",
                FACE_MESH_EVERY_N_FRAMES,
            )
            sys.exit(1)
        if FACE_MESH_TOP_N > FACE_PROCESS_TOP_N:
            logging.warning(
                "FACE_MESH_TOP_N (%d) > FACE_PROCESS_TOP_N (%d): "
                "se limitara mesh a %d.",
                FACE_MESH_TOP_N,
                FACE_PROCESS_TOP_N,
                FACE_PROCESS_TOP_N,
            )
        if FACE_MESH_EVERY_N_FRAMES > 10:
            logging.warning(
                "FACE_MESH_EVERY_N_FRAMES=%d es alto: la malla se vera entrecortada.",
                FACE_MESH_EVERY_N_FRAMES,
            )
        if not DISPLAY_IS_ENABLE:
            logging.warning(
                "ENABLE_FACEMESH=true pero DISPLAY_IS_ENABLE=false: "
                "FaceMesh es UX de overlay; el tick no correra sin display."
            )
        if INFERENCE_BACKEND == "pc":
            mesh_pc = FACEMESH_MODEL_PC
            if not os.path.isfile(mesh_pc):
                logging.critical(
                    "CONFIG ERROR: ENABLE_FACEMESH=true e INFERENCE_BACKEND=pc "
                    f"pero no existe FACEMESH_MODEL_PC: {mesh_pc}"
                )
                sys.exit(1)
            logging.info(
                "FaceMesh PC: %s (top %d desconocidos, cada %d frame(s)+hold)",
                mesh_pc,
                FACE_MESH_TOP_N,
                FACE_MESH_EVERY_N_FRAMES,
            )
        elif INFERENCE_BACKEND == "rk3568":
            mesh_rk = FACEMESH_MODEL_RK3568
            if not os.path.isfile(mesh_rk):
                logging.critical(
                    "CONFIG ERROR: ENABLE_FACEMESH=true e INFERENCE_BACKEND=rk3568 "
                    f"pero no existe FACEMESH_MODEL_RK3568: {mesh_rk}"
                )
                sys.exit(1)
            logging.info(
                "FaceMesh RK3568: %s (top %d desconocidos, cada %d frame(s)+hold)",
                mesh_rk,
                FACE_MESH_TOP_N,
                FACE_MESH_EVERY_N_FRAMES,
            )
        else:
            logging.info(
                "FaceMesh: ENABLE_FACEMESH=true pero INFERENCE_BACKEND=none "
                "(sin estimator)."
            )
    else:
        logging.info("FaceMesh: desactivado (ENABLE_FACEMESH=false)")

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
        if RETINAFACE_SCORE_DETECCION >= BYTETRACK_TRACK_THRESH:
            logging.warning(
                "RETINAFACE_SCORE_DETECCION (%.2f) >= BYTETRACK_TRACK_THRESH (%.2f): "
                "ByteTrack nunca recibira detecciones de baja confianza (RetinaFace ya "
                "las descarta antes de que lleguen); el segundo nivel de asociacion de "
                "ByteTrack (sostener tracks con detecciones debiles) queda sin uso. "
                "Bajar RETINAFACE_SCORE_DETECCION si se busca continuidad de tracking "
                "a traves de detecciones debiles.",
                RETINAFACE_SCORE_DETECCION,
                BYTETRACK_TRACK_THRESH,
            )
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

    if ENABLE_NOZZLE:
        if NOZZLE_PROCESS_TOP_N < 1:
            logging.critical("CONFIG ERROR: NOZZLE_PROCESS_TOP_N debe ser >= 1.")
            sys.exit(1)
        if NOZZLE_EVERY_N_FRAMES < 1:
            logging.critical(
                "CONFIG ERROR: NOZZLE_EVERY_N_FRAMES debe ser >= 1 (got %d).",
                NOZZLE_EVERY_N_FRAMES,
            )
            sys.exit(1)
        if NOZZLE_SCORE_DETECCION <= 0.0 or NOZZLE_SCORE_DETECCION > 1.0:
            logging.critical(
                "CONFIG ERROR: NOZZLE_SCORE_DETECCION debe estar en (0, 1] (got %.2f).",
                NOZZLE_SCORE_DETECCION,
            )
            sys.exit(1)
        if NOZZLE_NMS_IOU <= 0.0 or NOZZLE_NMS_IOU > 1.0:
            logging.critical(
                "CONFIG ERROR: NOZZLE_NMS_IOU debe estar en (0, 1] (got %.2f).",
                NOZZLE_NMS_IOU,
            )
            sys.exit(1)
        if NOZZLE_BYTETRACK_TRACK_THRESH <= 0.0 or NOZZLE_BYTETRACK_TRACK_THRESH > 1.0:
            logging.critical(
                "CONFIG ERROR: NOZZLE_BYTETRACK_TRACK_THRESH debe estar en (0, 1] "
                "(got %.2f).",
                NOZZLE_BYTETRACK_TRACK_THRESH,
            )
            sys.exit(1)
        if NOZZLE_BYTETRACK_MATCH_THRESH <= 0.0 or NOZZLE_BYTETRACK_MATCH_THRESH > 1.0:
            logging.critical(
                "CONFIG ERROR: NOZZLE_BYTETRACK_MATCH_THRESH debe estar en (0, 1] "
                "(got %.2f).",
                NOZZLE_BYTETRACK_MATCH_THRESH,
            )
            sys.exit(1)
        if NOZZLE_BYTETRACK_TRACK_BUFFER < 1:
            logging.critical(
                "CONFIG ERROR: NOZZLE_BYTETRACK_TRACK_BUFFER debe ser >= 1 (got %d).",
                NOZZLE_BYTETRACK_TRACK_BUFFER,
            )
            sys.exit(1)
        if NOZZLE_BYTETRACK_FRAME_RATE <= 0.0:
            logging.critical(
                "CONFIG ERROR: NOZZLE_BYTETRACK_FRAME_RATE debe ser > 0 (got %.2f).",
                NOZZLE_BYTETRACK_FRAME_RATE,
            )
            sys.exit(1)
        if NOZZLE_SCORE_DETECCION >= NOZZLE_BYTETRACK_TRACK_THRESH:
            logging.warning(
                "NOZZLE_SCORE_DETECCION (%.2f) >= NOZZLE_BYTETRACK_TRACK_THRESH (%.2f): "
                "pocas detecciones entraran al pool bajo de ByteTrack; bajar "
                "NOZZLE_SCORE_DETECCION si se busca continuidad con scores debiles.",
                NOZZLE_SCORE_DETECCION,
                NOZZLE_BYTETRACK_TRACK_THRESH,
            )
        if INFERENCE_BACKEND == "pc":
            nozzle_pc = NOZZLE_MODEL_PC
            if not os.path.isfile(nozzle_pc):
                logging.critical(
                    "CONFIG ERROR: ENABLE_NOZZLE=true e INFERENCE_BACKEND=pc "
                    f"pero no existe NOZZLE_MODEL_PC: {nozzle_pc}"
                )
                sys.exit(1)
            logging.info(
                "Nozzle PC: %s (top %d, score>=%.2f, cada %d frame(s), track_thresh=%.2f)",
                nozzle_pc,
                NOZZLE_PROCESS_TOP_N,
                NOZZLE_SCORE_DETECCION,
                NOZZLE_EVERY_N_FRAMES,
                NOZZLE_BYTETRACK_TRACK_THRESH,
            )
        elif INFERENCE_BACKEND == "rk3568":
            nozzle_rk = NOZZLE_MODEL_RK3568
            if not os.path.isfile(nozzle_rk):
                logging.critical(
                    "CONFIG ERROR: ENABLE_NOZZLE=true e INFERENCE_BACKEND=rk3568 "
                    f"pero no existe NOZZLE_MODEL_RK3568: {nozzle_rk}"
                )
                sys.exit(1)
            logging.info(
                "Nozzle RK3568: %s (top %d, score>=%.2f, cada %d frame(s), track_thresh=%.2f)",
                nozzle_rk,
                NOZZLE_PROCESS_TOP_N,
                NOZZLE_SCORE_DETECCION,
                NOZZLE_EVERY_N_FRAMES,
                NOZZLE_BYTETRACK_TRACK_THRESH,
            )
        else:
            logging.info(
                "Nozzle: ENABLE_NOZZLE=true pero INFERENCE_BACKEND=none (sin detector)."
            )
    else:
        logging.info("Nozzle: desactivado (ENABLE_NOZZLE=false)")


configure_logging()
