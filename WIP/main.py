"""
Punto de entrada del pipeline de biometria facial Edge (WIP).

Hasta ahora:
  - Carga configuracion desde configs.settings (modo RTSP/SNAP/USB, MAX_FPS, display, etc.)
    y valida parametros con validar_todo().
  - Arranca captura con utils.capture_cameras.CaptureCameras (hilo productor, warmup,
    limite de FPS y reconexion segun el modo).
  - Bucle principal: get_frame() -> log debug; opcional ventana OpenCV si
    DISPLAY_IS_ENABLE=true (env). Salir: Ctrl+C o tecla q en la ventana.

Sin modelos de inferencia todavia (deteccion/embeddings pendientes).

Ejemplo:
  cd WIP
  python main.py

  DISPLAY_IS_ENABLE=true CONFIG_MODO=USB python main.py

TODO:
    - RGA (Raster Graphic Acceleration Unit) para optimizar operaciones de OpenCV. 
    - Como resize y espacios de color 
    - Usar from rga import rga_context  # Módulo del wrapper de Rockchip
    - rga_context.init()
"""
import logging
import sys
import time
from pathlib import Path

import cv2

# import signal  # RK3568 + systemd: descomentar al portar a placa


def _project_root_with_utils(start_dir: Path) -> Path:
    """Sube desde start_dir hasta encontrar ./utils/ (RetinaFace + image_utils)."""
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro carpeta 'utils' con aux_tools_retinaface.py. "
        f"Origen de busqueda: {start_dir}"
    )


_WIP_DIR = Path(__file__).resolve().parent
ROOT = _project_root_with_utils(_WIP_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import settings as s  # noqa: E402
from utils.capture_cameras import CaptureCameras  # noqa: E402

# ejecutando_pipeline = True

# def manejador_apagado_linux(sig, frame):
#     global ejecutando_pipeline
#     logging.warning(f"Senal de parada detectada ({sig}). Cierre seguro...")
#     ejecutando_pipeline = False

# signal.signal(signal.SIGINT, manejador_apagado_linux)
# signal.signal(signal.SIGTERM, manejador_apagado_linux)


if __name__ == "__main__":
    s.validar_todo()

    ventana = "pipeline"
    if s.DISPLAY_IS_ENABLE:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        logging.info("Display activo (q en ventana para salir).")

    input_manager = CaptureCameras().start()

    logging.info("Pipeline de captura en marcha (sin modelos). Ctrl+C para salir.")

    try:
        while True:
            has_frame, frame = input_manager.get_frame()

            if has_frame and frame is not None:
                # TODO: deteccion, embeddings, etc.
                h, w = frame.shape[:2]
                logging.debug(f"Frame listo para procesar: {w}x{h}")

                if s.DISPLAY_IS_ENABLE:
                    cv2.imshow(ventana, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logging.info("Salida solicitada desde ventana (q).")
                        break
            else:
                if s.DISPLAY_IS_ENABLE:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logging.info("Salida solicitada desde ventana (q).")
                        break
                time.sleep(0.001)

    except KeyboardInterrupt:
        logging.warning("Interrupcion por teclado. Cerrando...")

    except Exception as error_critico:
        logging.critical(f"Fallo en el bucle principal: {error_critico}")

    finally:
        logging.info("Liberando hardware y sockets...")
        input_manager.stop()
        if s.DISPLAY_IS_ENABLE:
            cv2.destroyAllWindows()
        logging.info("Proceso terminado.")
        sys.exit(0)
