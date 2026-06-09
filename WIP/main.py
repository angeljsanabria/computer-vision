import logging
import sys
from pathlib import Path

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
if str(_WIP_DIR) not in sys.path:
    sys.path.insert(0, str(_WIP_DIR))

import settings as s  # noqa: E402
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

    input_manager = CaptureCameras().start()

    logging.info("Pipeline de captura en marcha (sin modelos). Ctrl+C para salir.")

    try:
        while True:
            has_frame, frame = input_manager.get_frame()

            if has_frame and frame is not None:
                # TODO: deteccion, embeddings, etc.
                h, w = frame.shape[:2]
                logging.debug(f"Frame listo para procesar: {w}x{h}")

    except KeyboardInterrupt:
        logging.warning("Interrupcion por teclado. Cerrando...")

    except Exception as error_critico:
        logging.critical(f"Fallo en el bucle principal: {error_critico}")

    finally:
        logging.info("Liberando hardware y sockets...")
        input_manager.stop()
        logging.info("Proceso terminado.")
        sys.exit(0)
