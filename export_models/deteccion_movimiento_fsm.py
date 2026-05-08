"""
Deteccion de movimiento (MOG2) + maquina de estados + RetinaFace ONNX opcional.

Copia ampliada de deteccion_movimiento.py con:

- Estados: IDLE, MOV_DETECTED, MOV_OUT (MOG2 sin pico en este frame),
  FACE_PROCESSED, FACE_OUT (cara no vista este frame; distinto del timeout 10s),
  FACE_PROCESSED_TIMEOUT.
- Timeout --timeout_seg hacia IDLE por MOG2: solo aplica en MOV_DETECTED / MOV_OUT
  (persona quieta con cara en FACE_* no se expulsa por falta de movimiento de fondo).
- Sin cara durante --timeout_seg desde ultima cara: FACE_OUT -> FACE_PROCESSED_TIMEOUT -> IDLE.

Inferencia en MOV_*, FACE_PROCESSED, FACE_OUT. Sin modelo (--sin_modelo)
permite probar la FSM en PC sin onnxruntime.

Ejemplo:
  python export_models/deteccion_movimiento_fsm.py
  python export_models/deteccion_movimiento_fsm.py --timeout_seg 10 --ventana
  python export_models/deteccion_movimiento_fsm.py --sin_log_sensores
"""
from __future__ import annotations

import argparse
import sys
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# --- repo root (utils) ---
def _project_root_with_utils(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro carpeta 'utils' con aux_tools_retinaface.py. "
        "Ejecuta desde el repo computer-vision."
    )


_EXPORT_DIR = Path(__file__).resolve().parent
ROOT = _project_root_with_utils(_EXPORT_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aux_tools_retinaface import (  # noqa: E402
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr  # noqa: E402


class FlowState(str, Enum):
    IDLE = "IDLE"
    MOV_DETECTED = "MOV_DETECTED"
    MOV_OUT = "MOV_OUT"
    FACE_PROCESSED = "FACE_PROCESSED"
    FACE_OUT = "FACE_OUT"
    FACE_PROCESSED_TIMEOUT = "FACE_PROCESSED_TIMEOUT"


RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2


def _preprocess_for_onnx(inp0_meta, canvas_bgr: np.ndarray) -> np.ndarray:
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape

    if len(dims) == 4 and dims[1] == 3:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
        return feed
    if len(dims) == 4 and dims[-1] == 3:
        x32 -= RETINAFACE_MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)
    chw = np.transpose(x32, (2, 0, 1))
    feed = np.expand_dims(chw, axis=0)
    feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
    return feed


def retina_infer(
    session,
    input_name: str,
    inp0_meta,
    frame_bgr: np.ndarray,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    letterbox_img, lb_meta = letterbox_bgr(
        frame_bgr,
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
        RETINAFACE_LETTERBOX_FILL,
    )
    tensor = _preprocess_for_onnx(inp0_meta, letterbox_img)
    ort_outputs = session.run(None, {input_name: tensor})
    return retinaface_dets_desde_rknn_outputs(
        list(ort_outputs),
        img_width=w,
        img_height=h,
        aspect_ratio=lb_meta.aspect_ratio,
        offset_x=lb_meta.offset_x,
        offset_y=lb_meta.offset_y,
        score_deteccion=RETINAFACE_SCORE_DETECCION,
        score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
        log_priors=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOG2 + FSM + RetinaFace ONNX (opcional)."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ancho", type=int, default=320)
    parser.add_argument("--alto", type=int, default=240)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--var_threshold", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--movimiento_pixeles", type=int, default=1000)
    parser.add_argument(
        "--timeout_seg",
        type=float,
        default=10.0,
        help=(
            "Segundos: (1) sin movimiento MOG2 solo en MOV_* -> IDLE; "
            "(2) sin cara en FACE_OUT -> FACE_PROCESSED_TIMEOUT"
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT / "Retinaface-Models" / "RetinaFace_mobile320.onnx"),
    )
    parser.add_argument(
        "--sin_modelo",
        action="store_true",
        help="No cargar ONNX; solo MOG2 y estados (prueba de FSM).",
    )
    parser.add_argument(
        "--ventana",
        action="store_true",
        help="Muestra cv2.imshow con estado y frame.",
    )
    parser.add_argument(
        "--sin_log_sensores",
        action="store_true",
        help="No imprime lineas MOG2/RetinaFace por frame (solo transiciones FSM).",
    )
    args = parser.parse_args()
    log_sensores = not args.sin_log_sensores

    periodo_s = 1.0 / max(args.fps, 0.1)
    procesar_wh = (args.ancho, args.alto)
    T = float(args.timeout_seg)

    session = None
    input_name = ""
    inp0 = None
    if not args.sin_modelo:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit(
                "Instala onnxruntime o usa --sin_modelo. " + str(e)
            ) from e
        if not Path(args.model_path).is_file():
            raise SystemExit("No existe ONNX: " + args.model_path)
        session = ort.InferenceSession(
            args.model_path, providers=["CPUExecutionProvider"]
        )
        inp0 = session.get_inputs()[0]
        input_name = inp0.name

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            "No se abrio la camara {}. Prueba otro --camera.".format(args.camera)
        )

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False,
    )

    print("Calibrando fondo...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if ret:
            fgbg.apply(
                cv2.resize(frame, procesar_wh, interpolation=cv2.INTER_AREA),
                learningRate=0.5,
            )
        else:
            break

    state: FlowState = FlowState.IDLE
    t_ultimo_mov: float | None = None
    t_ultima_cara: float | None = None

    print("Listo. Estados: {}. Ctrl+C salir.".format(", ".join(s.value for s in FlowState)))

    try:
        while True:
            t_loop = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                print("Fin de captura.")
                break

            now = time.monotonic()
            small_frame = cv2.resize(
                frame, procesar_wh, interpolation=cv2.INTER_AREA
            )
            mask = fgbg.apply(small_frame)
            pixel_count = int(cv2.countNonZero(mask))
            hay_mov = pixel_count > args.movimiento_pixeles

            if log_sensores:
                if hay_mov:
                    print(
                        "[MOG2] MOV_DETECTED pixels={} umbral={}".format(
                            pixel_count, args.movimiento_pixeles
                        )
                    )
                else:
                    print(
                        "[MOG2] NOT_MOV pixels={} umbral={}".format(
                            pixel_count, args.movimiento_pixeles
                        )
                    )

            if hay_mov:
                t_ultimo_mov = now

            sin_mov_mog2 = (
                t_ultimo_mov is not None and (now - t_ultimo_mov) >= T
            )

            # 1) Sin movimiento MOG2 T seg -> IDLE solo desde fase movimiento (no desde FACE_*)
            if sin_mov_mog2 and state in (
                FlowState.MOV_DETECTED,
                FlowState.MOV_OUT,
            ):
                s_antes = state
                state = FlowState.IDLE
                t_ultima_cara = None
                print(
                    "[FSM] {} -> IDLE (timeout {:.1f} s sin movimiento MOG2)".format(
                        s_antes.value, T
                    )
                )

            # 2) Transiciones por movimiento (MOG2)
            s_antes_mog2 = state
            if state == FlowState.IDLE:
                if hay_mov:
                    state = FlowState.MOV_DETECTED
            elif state == FlowState.MOV_DETECTED:
                if not hay_mov:
                    state = FlowState.MOV_OUT
            elif state == FlowState.MOV_OUT:
                if hay_mov:
                    state = FlowState.MOV_DETECTED
            # FACE_*: no cambian por MOG2 salvo reset por sin_mov arriba

            if state != s_antes_mog2:
                print(
                    "[FSM] {} -> {} (MOG2)".format(
                        s_antes_mog2.value, state.value
                    )
                )

            # 3) Inferencia facial
            hay_cara = False
            ejecutar_infer = state in (
                FlowState.MOV_DETECTED,
                FlowState.MOV_OUT,
                FlowState.FACE_PROCESSED,
                FlowState.FACE_OUT,
            ) and session is not None

            s_antes_infer = state
            if ejecutar_infer:
                dets = retina_infer(session, input_name, inp0, frame)
                hay_cara = dets.shape[0] > 0
                if log_sensores:
                    if hay_cara:
                        print(
                            "[RetinaFace] FACE_DETECTED num_dets={}".format(
                                int(dets.shape[0])
                            )
                        )
                    else:
                        print("[RetinaFace] NOT_FACE_IN_IMG")
                if hay_cara:
                    t_ultima_cara = now
                    t_ultimo_mov = now
                    state = FlowState.FACE_PROCESSED
                elif state == FlowState.FACE_PROCESSED:
                    state = FlowState.FACE_OUT
                elif state == FlowState.FACE_OUT:
                    if t_ultima_cara is not None and (now - t_ultima_cara) >= T:
                        state = FlowState.FACE_PROCESSED_TIMEOUT

            if ejecutar_infer and state != s_antes_infer:
                print(
                    "[FSM] {} -> {} (RetinaFace)".format(
                        s_antes_infer.value, state.value
                    )
                )

            # FACE_PROCESSED_TIMEOUT -> IDLE en el mismo tick
            if state == FlowState.FACE_PROCESSED_TIMEOUT:
                print(
                    "[FSM] {} -> IDLE (sin cara durante {:.1f} s)".format(
                        FlowState.FACE_PROCESSED_TIMEOUT.value, T
                    )
                )
                state = FlowState.IDLE
                t_ultima_cara = None

            if args.ventana:
                vis = frame.copy()
                msg = "estado: {}".format(state.value)
                cv2.putText(
                    vis,
                    msg,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("deteccion_movimiento_fsm", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.monotonic() - t_loop
            espera = periodo_s - elapsed
            if espera > 0:
                time.sleep(espera)

    except KeyboardInterrupt:
        print("Interrupcion.")
    finally:
        cap.release()
        if args.ventana:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
