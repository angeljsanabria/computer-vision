import os
import sys
import logging

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

# 1.2 Detalles de Captura
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1"))
REINTENTO_SEG = float(os.getenv("REINTENTO_SEG", "10"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "10"))
LOG_CADA_N_FRAMES = int(os.getenv("LOG_CADA_N_FRAMES", "10"))

# 2. HARDWARE LOCAL (CAMARA USB)
USB_INDEX = int(os.getenv("USB_DEVICE_INDEX", 0))

# 3. CONFIGURACIONES Camara IP
_user = os.getenv("IP_CAM_USER", "angelcam")
_pass = os.getenv("IP_CAM_PASS", "angelCamara")
_host_ip = os.getenv("IP_CAM", "192.168.1.16")  # info dispositivo; info red

# 3.1 RTSP
_port = os.getenv("IP_CAM_RTSP_PORT", "554")   # info dispositivo; info avanzada
_route_rtsp_quality_low = os.getenv("IP_CAM_RTSP_ROUTE", "Preview_01_main")
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

# 4. RUTAS DE LOS MODELOS RKNN
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
