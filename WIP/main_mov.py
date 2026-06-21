"""
Pipeline edge WIP: captura + MOG2 + FSM.

Responsabilidad de este archivo: orquestar el bucle por frame. La logica de
movimiento y estados vive en ``mov_detect/``; captura en ``utils/``;
inferencia (RetinaFace via ``inference/``, MobileFaceNet pendiente).

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

import cv2

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
from utils.capture_cameras import CaptureCameras  # noqa: E402

ROOT = project_root(_WIP_DIR)
WINDOW_NAME = "pipeline_mov"

# Fase keep-alive 0..2 -> ".", "..", "..." (se reinicia en la 4. entrada).
_keep_alive_phase = 0

# ---------------------------------------------------------------------------
# Helpers locales (solo orquestacion / UI de depuracion).
#
# No contienen logica de negocio: MOG2, FSM e inferencia viven en mov_detect/
# e inference/. Estas funciones existen para mantener main() legible y evitar
# duplicar codigo de logging, overlay y tecla 'q' en el bucle.
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
    Inferencia RetinaFace + tick_face cuando la FSM lo indica.

    ``face`` viene de ``build_face_detector()`` (None si INFERENCE_BACKEND=none).
    """
    if not fsm_out.run_face_detector or face is None:
        return None, fsm_out

    dets = face.detect(frame)
    estado_antes = fsm.state
    fsm_out = fsm.tick_face(hay_cara=dets.has_faces, now=now)
    _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
    _log_transitions(fsm_out.transitions)
    return dets, fsm_out


def _tick_keep_alive() -> str:
    """
    Avanza el indicador keep-alive (0..2) y devuelve '.', '..' o '...'.

    En Python no hay ``static`` local: el contador vive a nivel modulo porque
    debe persistir entre llamadas a la funcion.
    """
    global _keep_alive_phase
    dots = "." * (_keep_alive_phase + 1)
    _keep_alive_phase = (_keep_alive_phase + 1) % 3
    return dots


def _draw_overlay(
    frame,
    fsm_out: FsmTickResult,
    mov: MotionResult,
    dets: FaceDetections | None = None,
):
    """
    Devuelve copia del frame con texto de depuracion (estado FSM + MOG2).

    Si ``dets`` tiene caras, dibuja bbox rojas. Solo para DISPLAY_IS_ENABLE.
    """
    vis = frame.copy()
    _, w = vis.shape[:2]
    alive = _tick_keep_alive()
    (tw, th), _ = cv2.getTextSize(alive, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        vis,
        alive,
        (w - tw - 8, th + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )
    if dets is not None and dets.has_faces:
        for row in dets.dets:
            x1, y1, x2, y2 = map(int, row[:4])
            score = float(row[4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
    cv2.putText(
        vis,
        f"estado: {fsm_out.state.value}",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"MOG2 px={mov.pixel_count} mov={mov.hay_mov}",
        (8, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )
    return vis


def _quit_requested() -> bool:
    """True si el usuario pulso 'q' en la ventana OpenCV (requiere display activo)."""
    return cv2.waitKey(1) & 0xFF == ord("q")


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

    if s.DISPLAY_IS_ENABLE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        logging.info("Display activo (q en ventana para salir).")

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

                # Proximo paso: embedder cuando fsm_out.run_embedding

                if s.DISPLAY_IS_ENABLE:
                    cv2.imshow(
                        WINDOW_NAME,
                        _draw_overlay(frame, fsm_out, mov, dets),
                    )
                    if _quit_requested():
                        logging.info("Salida solicitada desde ventana (q).")
                        break
            else:
                if s.DISPLAY_IS_ENABLE and _quit_requested():
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
        capture.stop()
        if s.DISPLAY_IS_ENABLE:
            cv2.destroyAllWindows()
        logging.info("Proceso terminado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
