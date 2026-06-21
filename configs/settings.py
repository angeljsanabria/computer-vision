import os
import sys
import logging

from configs.paths import resolve_repo_path

# Configuracion de logs para produccion
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 1. CONFIGURACIONES GENERALES
# 1.1 Captura
MODO = os.getenv("CONFIG_MODO", "USB").upper()     # RTSP, SNAP, USB
MAX_FPS = float(os.getenv("MAX_FPS", 2.0))
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", 15))
DISPLAY_IS_ENABLE = (
    os.getenv("DISPLAY_IS_ENABLE", "true").lower() == "true"
)
# Cuantas caras rankeadas procesar (1=mejor, 2=mejor+siguiente, ...).
FACE_PROCESS_TOP_N = int(os.getenv("FACE_PROCESS_TOP_N", 2))

# 1.2 Detalles de Captura
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1"))
CAP_FRAME_WIDTH = int(os.getenv("CAP_FRAME_WIDTH", 640))   #  High 2560    Medium 1080     Low 640
CAP_FRAME_HEIGHT = int(os.getenv("CAP_FRAME_HEIGHT", 480))  #  High 1920    Medium 720      Low 480
REINTENTO_SEG = float(os.getenv("REINTENTO_SEG", "10"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "10"))
LOG_CADA_N_FRAMES = int(os.getenv("LOG_CADA_N_FRAMES", "10"))

# 1.3 Procesamiento de imagen (RGA RK3568; legacy OpenCV por defecto)
USE_RGA = os.getenv("USE_RGA", "false").lower() == "true"

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

# 6. INFERENCIA (RetinaFace; MobileFaceNet pendiente)
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "pc").lower()  # "pc", "rk3568"
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

def retinaface_model_pc_path() -> str:
    """Ruta absoluta al ONNX RetinaFace (defecto: models_onnx/RetinaFace_mobile320.onnx)."""
    return str(resolve_repo_path(RETINAFACE_MODEL_PC))


def retinaface_model_rk3568_path() -> str:
    """Ruta absoluta al RKNN RetinaFace (defecto: models/RetinaFace_mobile320.rknn)."""
    return str(resolve_repo_path(RETINAFACE_MODEL_RK3568))

# 7. RUTAS DE LOS MODELOS RKNN (legacy / futuro embed)
#RETINAFACE_MODEL = os.getenv("RETINAFACE_PATH", "./models/retinaface.rknn")
#MOBILEFACENET_MODEL = os.getenv("MOBILEFACENET_PATH", "./models/mobilefacenet.rknn")


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

    if INFERENCE_BACKEND == "rk3568":
        rk_path = retinaface_model_rk3568_path()
        if not os.path.isfile(rk_path):
            logging.critical(
                "CONFIG ERROR: INFERENCE_BACKEND=rk3568 pero no existe "
                f"RETINAFACE_MODEL_RK3568: {rk_path}"
            )
            sys.exit(1)
        logging.info("RetinaFace RK3568: %s", rk_path)

    if FACE_PROCESS_TOP_N < 1:
        logging.critical("CONFIG ERROR: FACE_PROCESS_TOP_N debe ser >= 1.")
        sys.exit(1)
    logging.info(
        "Caras a procesar: top %d (score >= RETINAFACE_SCORE_DETECCION=%.2f)",
        FACE_PROCESS_TOP_N,
        RETINAFACE_SCORE_DETECCION,
    )
