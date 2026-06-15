"""
Pipeline WIP: captura + MOG2 + FSM (sin RetinaFace aun).

Flujo:
  1. CaptureCameras.start()
  2. MOG2 warmup: primer frame x N applies (solo al inicio)
  3. Bucle: MOG2 + FSM con frames nuevos del stream

Ejemplo:
  cd WIP
  python main_mov.py

  DISPLAY_IS_ENABLE=true CONFIG_MODO=USB python main_mov.py
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
from mov_detect import (  # noqa: E402
    FlowState,
    MotionFaceFsm,
    Mog2MotionSensor,
    config_from_settings,
)
from utils.capture_cameras import CaptureCameras  # noqa: E402

# def manejador_apagado_linux(sig, frame):
#     global ejecutando_pipeline
#     logging.warning(f"Senal de parada detectada ({sig}). Cierre seguro...")
#     ejecutando_pipeline = False

# signal.signal(signal.SIGINT, manejador_apagado_linux)
# signal.signal(signal.SIGTERM, manejador_apagado_linux)
def _log_transitions(transitions: tuple[str, ...]) -> None:
    for msg in transitions:
        logging.info(msg)


def _sync_umbral_mog2(
    motion: Mog2MotionSensor,
    estado_antes,
    estado_despues,
) -> None:
    if estado_despues == estado_antes:
        return
    if estado_despues == FlowState.IDLE:
        motion.set_umbral_idle()
    elif estado_antes == FlowState.IDLE:
        motion.set_umbral_activo()


def _log_mog2(mov, umbral: int) -> None:
    if mov.hay_mov:
        logging.info(
            "[MOG2] MOV_DETECTED pixels=%d umbral=%d",
            mov.pixel_count,
            umbral,
        )
    else:
        logging.info(
            "[MOG2] NOT_MOV pixels=%d umbral=%d",
            mov.pixel_count,
            umbral,
        )


if __name__ == "__main__":
    s.validar_todo()

    mog2_cfg, fsm_cfg = config_from_settings()
    motion = Mog2MotionSensor(mog2_cfg)
    fsm = MotionFaceFsm(fsm_cfg)

    ventana = "pipeline_mov"
    if s.DISPLAY_IS_ENABLE:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        logging.info("Display activo (q en ventana para salir).")

    capture = CaptureCameras().start()

    motion.warmup_from_first_frame(
        capture.get_frame,
        n_frames=mog2_cfg.warmup_frames,
        timeout_s=s.MOG2_WARMUP_TIMEOUT_S,
    )

    logging.info(
        "Pipeline MOG2+FSM en marcha (RetinaFace pendiente). Ctrl+C para salir."
    )

    try:
        while True:
            has_frame, frame = capture.get_frame()

            if has_frame and frame is not None:
                now = time.monotonic()
                estado_antes = fsm.state
                mov = motion.evaluate(frame)
                _log_mog2(mov, motion.umbral_pixeles)

                fsm_out = fsm.tick_motion(hay_mov=mov.hay_mov, now=now)
                _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
                _log_transitions(fsm_out.transitions)

                # TODO: face = build_face_detector(config)
                # if fsm_out.run_face_detector and face is not None:
                #     dets = face.detect(frame)
                #     estado_antes = fsm.state
                #     fsm_out = fsm.tick_face(hay_cara=dets.has_faces, now=now)
                #     _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
                #     _log_transitions(fsm_out.transitions)
                # if fsm_out.run_embedding and dets is not None:
                #     identity.process(frame, dets)

                if s.DISPLAY_IS_ENABLE:
                    vis = frame.copy()
                    cv2.putText(
                        vis,
                        "estado: {}".format(fsm_out.state.value),
                        (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        vis,
                        "MOG2 px={} mov={}".format(mov.pixel_count, mov.hay_mov),
                        (8, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                    )
                    cv2.imshow(ventana, vis)
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
        logging.critical("Fallo en el bucle principal: %s", error_critico)

    finally:
        logging.info("Liberando hardware y sockets...")
        capture.stop()
        if s.DISPLAY_IS_ENABLE:
            cv2.destroyAllWindows()
        logging.info("Proceso terminado.")
        sys.exit(0)
