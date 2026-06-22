"""
Pipeline edge WIP: captura + MOG2 + FSM.

Responsabilidad de este archivo: orquestar el bucle por frame. La logica de
movimiento y estados vive en ``mov_detect/``; captura en ``utils/``;
inferencia (RetinaFace via ``inference/``, MobileFaceNet pendiente); UI en ``ui/``.

Flujo por frame:
  1. ``CaptureCameras.get_frame()`` — frame BGR canonico.
  2. ``Mog2MotionSensor.evaluate()`` — movimiento sobre frame reducido.
  3. ``MotionFaceFsm.tick_motion()`` — transiciones IDLE / MOV_*.
  4. ``_sync_umbral_mog2()`` — histéresis de umbral MOG2 segun estado FSM.
  5. Si ``run_face_detector``: ``build_face_detector().detect()`` + ``tick_face()``.

Estados FSM (resumen):
  IDLE          — sin inferencia facial.
  MOV_DETECTED  — MOG2 supero umbral.
  MOV_OUT       — MOG2 bajo umbral dentro de sesion activa.
  FACE_*        — RetinaFace activo (cuando INFERENCE_BACKEND != none).

Variables de entorno utiles (ver ``configs/settings.py``):
  CONFIG_MODO          — USB | RTSP | SNAP
  DISPLAY_IS_ENABLE    — true/false (overlay OpenCV)
  MOG2_* / FSM_*       — umbrales y timeouts
  INFERENCE_BACKEND    — none | pc | rk3568 (factory en ``inference/``)
  FACE_PROCESS_TOP_N     — 1=mejor cara, 2=mejor+siguiente, ...

Ejemplos:
  cd WIP
  python main_mov.py

  INFERENCE_BACKEND=pc DISPLAY_IS_ENABLE=true CONFIG_MODO=USB python main_mov.py
  INFERENCE_BACKEND=none python main_mov.py

Despliegue RK3568: registrar manejadores SIGINT/SIGTERM para cierre limpio
con systemd (ver ``main.py`` / comentarios historicos en el repo).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Bootstrap: permite ``python main_mov.py`` desde WIP/ sin instalar el paquete.
# Sube directorios hasta encontrar configs/settings.py y agrega esa raiz a sys.path.
_WIP_DIR = Path(__file__).resolve().parent
for _candidate in [_WIP_DIR, *_WIP_DIR.parents]:
    if (_candidate / "configs" / "settings.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError(
        "No se encontro raiz del repo (configs/settings.py). "
        f"Origen: {_WIP_DIR}"
    )

from configs import settings as s  # noqa: E402
from configs.paths import project_root  # noqa: E402
from mov_detect import (  # noqa: E402
    FlowState,
    MotionFaceFsm,
    Mog2MotionSensor,
    MotionResult,
    config_from_settings,
)
from mov_detect.types import FsmTickResult  # noqa: E402
from inference import FaceDetector, build_face_detector  # noqa: E402
from inference.types import FaceDetections  # noqa: E402
from inference.retinaface.select_best import mejores_caras  # noqa: E402
from ui import FrameView, PipelineDisplay  # noqa: E402
from utils.capture_cameras import CaptureCameras  # noqa: E402

ROOT = project_root(_WIP_DIR)

# ---------------------------------------------------------------------------
# Helpers locales (orquestacion del pipeline por etapas).
#
# MOG2, FSM e inferencia viven en mov_detect/ e inference/. La UI vive en ui/.
# ---------------------------------------------------------------------------


def _log_transitions(transitions: tuple[str, ...]) -> None:
    """Escribe en log las transiciones FSM que devolvio tick_motion/tick_face."""
    for msg in transitions:
        logging.info(msg)


def _sync_umbral_mog2(
    motion: Mog2MotionSensor,
    estado_antes: FlowState,
    estado_despues: FlowState,
) -> None:
    """
    Sincroniza umbral MOG2 con el estado FSM (histéresis).

    En IDLE usa el umbral base (menos sensible). Al salir de IDLE baja el
    umbral (mas sensible) para no perder actividad durante MOV_* / FACE_*.
    Debe llamarse despues de cada tick que cambie ``fsm.state``.
    """
    if estado_despues == estado_antes:
        return
    if estado_despues == FlowState.IDLE:
        motion.set_umbral_idle()
    elif estado_antes == FlowState.IDLE:
        motion.set_umbral_activo()


def _log_mog2(mov: MotionResult, umbral: int) -> None:
    """Log de una lectura MOG2 (pixeles en mascara vs umbral activo)."""
    tag = "MOV_DETECTED" if mov.hay_mov else "NOT_MOV"
    logging.info("[MOG2] %s pixels=%d umbral=%d", tag, mov.pixel_count, umbral)


def _tick_mog2_fsm(
    motion: Mog2MotionSensor,
    fsm: MotionFaceFsm,
    frame,
    now: float,
) -> tuple[MotionResult, FsmTickResult]:
    """
    Un ciclo MOG2 + FSM por frame (fase actual del pipeline).

    Encapsula evaluate -> tick_motion -> sync umbral -> log. RetinaFace y
    embed iran despues de este bloque, usando ``fsm_out.run_face_detector`` y
    ``fsm_out.run_embedding`` (no dentro de esta funcion).
    """
    estado_antes = fsm.state
    mov = motion.evaluate(frame)
    motion.log_motion_if_changed(mov)
    #_log_mog2(mov, motion.umbral_pixeles)  # ver los de movimiento en cada frame

    fsm_out = fsm.tick_motion(hay_mov=mov.hay_mov, now=now)
    _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
    _log_transitions(fsm_out.transitions)
    return mov, fsm_out


def _tick_retinaface_if_needed(
    face: FaceDetector | None,
    fsm: MotionFaceFsm,
    motion: Mog2MotionSensor,
    frame,
    now: float,
    fsm_out: FsmTickResult,
) -> tuple[FaceDetections | None, FsmTickResult]:
    """
    RetinaFace + ranking + tick_face cuando la FSM lo indica.

    Devuelve solo las caras utiles (top-N rankeadas) y el estado FSM.
    La FSM usa ``hay_cara`` sobre todas las detecciones del modelo, no solo las filtradas.
    """
    if not fsm_out.run_face_detector or face is None:
        return None, fsm_out

    raw = face.detect(frame)
    dets = mejores_caras(raw, top_n=s.FACE_PROCESS_TOP_N)
    estado_antes = fsm.state
    fsm_out = fsm.tick_face(hay_cara=raw.has_faces, now=now)
    _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
    _log_transitions(fsm_out.transitions)
    return dets, fsm_out


def _release_face_detector(face: FaceDetector | None) -> None:
    """Libera runtime del detector si expone release() (p. ej. RKNNLite en RK3568)."""
    if face is None:
        return
    release = getattr(face, "release", None)
    if callable(release):
        release()


def main() -> int:
    """
    Punto de entrada: valida config, warmup MOG2, bucle por frame, cleanup.

    Retorna 0 si termino bien; 1 si hubo excepcion no controlada en el bucle.
    """
    s.validar_todo()
    logging.info("Repo root: %s", ROOT)

    mog2_cfg, fsm_cfg = config_from_settings()
    motion = Mog2MotionSensor(mog2_cfg)
    fsm = MotionFaceFsm(fsm_cfg)
    face = build_face_detector()
    if face is not None:
        logging.info("RetinaFace activo (backend=%s)", s.INFERENCE_BACKEND)
    else:
        logging.info(
            "RetinaFace desactivado (INFERENCE_BACKEND=%s)", s.INFERENCE_BACKEND
        )

    display = PipelineDisplay.from_settings()
    display.setup()

    capture = CaptureCameras().start()
    try:
        motion.warmup_from_first_frame(
            capture.get_frame,
            n_frames=mog2_cfg.warmup_frames,
            timeout_s=s.MOG2_WARMUP_TIMEOUT_S,
        )

        logging.info(
            "Pipeline MOG2+FSM+RetinaFace en marcha. Ctrl+C para salir."
        )
        #motion.reset_motion_log()

        while True:
            has_frame, frame = capture.get_frame()

            if has_frame and frame is not None:
                now = time.monotonic()
                mov, fsm_out = _tick_mog2_fsm(motion, fsm, frame, now)
                dets, fsm_out = _tick_retinaface_if_needed(
                    face, fsm, motion, frame, now, fsm_out
                )

                # Proximo paso: embedder sobre dets.dets (ya top-N rankeadas)

                view = FrameView(mov=mov, fsm=fsm_out, dets=dets)
                display.show(frame, view)
                if display.poll_quit():
                    logging.info("Salida solicitada desde ventana (q).")
                    break
            else:
                if display.poll_quit():
                    logging.info("Salida solicitada desde ventana (q).")
                    break
                time.sleep(0.001)

    except KeyboardInterrupt:
        logging.warning("Interrupcion por teclado. Cerrando...")
    except Exception as exc:
        logging.critical("Fallo en el bucle principal: %s", exc, exc_info=True)
        return 1
    finally:
        logging.info("Liberando hardware y sockets...")
        _release_face_detector(face)
        capture.stop()
        display.teardown()
        logging.info("Proceso terminado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
