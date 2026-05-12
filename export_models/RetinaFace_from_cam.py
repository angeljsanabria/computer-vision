"""
RetinaFace con ONNX en PC (OnnxRuntime) desde camara USB.

Misma inferencia y postproceso que RetinaFace_from_img_onnx / RetinaFace_from_cam
(letterbox, tensor segun firma ONNX, retinaface_dets_desde_rknn_outputs).

Captura alineada con RetinaFace_from_ip_cam.py: buffer corto, hilo de ultimo frame,
limite MAX_FPS_ANALISIS, log periodico, guardado con intervalo, reconexion si falla
el frame (solo cambia la fuente: USB en lugar de RTSP).

Ejemplo:
  python export_models/RetinaFace_from_cam.py --display
  python export_models/RetinaFace_from_cam.py --camera 1 --no-save
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit(
        "No se pudo importar onnxruntime.\n"
        f"  Python usado: {sys.executable}\n"
        "  Instala en ESE intérprete (con el venv activado): pip install onnxruntime\n"
        f"  Detalle: {e!r}"
    ) from e


def _project_root_with_utils(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro carpeta 'utils' con aux_tools_retinaface.py. "
        f"Origen de busqueda: {start_dir}"
    )


ROOT = _project_root_with_utils(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aux_tools_retinaface import (  # noqa: E402
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr  # noqa: E402

# --- Misma politica de captura que RetinaFace_from_ip_cam.py (fuente = USB) ---
USAR_HILO_CAPTURA = True
TAMANO_BUFFER_CAMARA = 1
MAX_FPS_ANALISIS = 2.0
LOG_CADA_CAPS = 10
TIME_SAVE_DETECTION = 3 * 60
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_retinaface_usb"
REINTENTO_CAPTURA_SEG = 10

RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2


def _preprocess_for_onnx(inp0_meta: ort.NodeArg, canvas_bgr: np.ndarray) -> np.ndarray:
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape

    nchw = False
    nhwc = False
    if len(dims) == 4 and dims[1] == 3:
        nchw = True
    elif len(dims) == 4 and dims[-1] == 3:
        nhwc = True
    elif len(dims) == 4 and dims[1] == RETINAFACE_INPUT_WIDTH:
        nhwc = True
    else:
        nchw = True

    if nhwc:
        x32 -= RETINAFACE_MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)

    if nchw:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
        return feed

    raise RuntimeError("Forma de entrada ONNX no soportada: " + repr(dims))


def configurar_buffer_camara(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, TAMANO_BUFFER_CAMARA)
    except Exception:
        pass


def log_fps_analisis(frame_count: int, t0_tick: int, frame: np.ndarray) -> None:
    if frame_count % LOG_CADA_CAPS != 0:
        return
    ticks = cv2.getTickCount() - t0_tick
    dt = ticks / cv2.getTickFrequency()
    fps = frame_count / dt if dt > 0 else 0.0
    h, w = frame.shape[:2]
    print(f"[LOG] frame={frame_count} size={w}x{h} fps_aprox={fps:.2f}")


def construir_file_path_dia() -> str:
    dd_mm = time.strftime("%d_%m")
    file_name_img = f"{FILE_BASE_NAME_IMG}_{dd_mm}.jpg"
    return os.path.join(FILE_DIR, file_name_img)


def abrir_usb_con_calentamiento(device_index: int) -> cv2.VideoCapture | None:
    """Abre V4L/USB y descarta frames iniciales (misma idea que calentar RTSP)."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        return None

    for _ in range(25):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            configurar_buffer_camara(cap)
            return cap

    cap.release()
    return None


def esperar_primer_frame_grabber(
    grabber: "UltimoFrameCamara", timeout_seg: float = 2.0
) -> bool:
    t_ini = time.time()
    while (time.time() - t_ini) < timeout_seg:
        ok, _ = grabber.read_copy()
        if ok:
            return True
        time.sleep(0.05)
    return False


class UltimoFrameCamara:
    """
    Un solo hilo llama a cap.read(); el bucle principal copia el ultimo frame.
    Evita que mientras ONNX infiere se acumulen frames viejos en el buffer.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                with self._lock:
                    self._frame = None
                time.sleep(0.001)

    def read_copy(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def main() -> None:
    _repo = ROOT
    default_onnx = _repo / "Retinaface-Models" / "RetinaFace_mobile320.onnx"

    parser = argparse.ArgumentParser(
        description="RetinaFace ONNX en PC desde camara USB (captura como RetinaFace_from_ip_cam)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_onnx),
        help="Ruta al .onnx",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice de camara USB (0 por defecto).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Ventana OpenCV (q o ESC para salir).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar imagenes con detecciones.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="OnnxRuntime providers separados por coma.",
    )
    args = parser.parse_args()

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el ONNX: {args.model_path}")

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    session = ort.InferenceSession(args.model_path, providers=providers)
    inp0 = session.get_inputs()[0]
    input_name = inp0.name

    cap = abrir_usb_con_calentamiento(args.camera)
    while cap is None:
        print(
            f"[RETRY] No se pudo abrir/calentar USB cam {args.camera}. "
            f"Reintento en {REINTENTO_CAPTURA_SEG}s..."
        )
        time.sleep(REINTENTO_CAPTURA_SEG)
        cap = abrir_usb_con_calentamiento(args.camera)

    grabber: UltimoFrameCamara | None = None
    if USAR_HILO_CAPTURA:
        grabber = UltimoFrameCamara(cap)
        grabber.start()
        if not esperar_primer_frame_grabber(grabber):
            grabber.stop()
            cap.release()
            raise SystemExit(
                "Camara USB abierta pero el hilo no recibio el primer frame. "
                "Revisar permisos / indice --camera."
            )
        print("Captura en hilo auxiliar activa (menos latencia por buffer).")

    if args.display:
        print("Camara USB lista. Modo display (q o ESC para salir).")
    else:
        print("Camara USB lista. Sin display (Ctrl+C para salir).")

    periodo_analisis_ticks = (
        int(cv2.getTickFrequency() / MAX_FPS_ANALISIS)
        if MAX_FPS_ANALISIS > 0
        else 0
    )
    next_due = cv2.getTickCount()
    frame_count = 0
    t0_tick = cv2.getTickCount()
    save_interval_ticks = int(TIME_SAVE_DETECTION * cv2.getTickFrequency())
    last_save_tick = cv2.getTickCount()

    try:
        while True:
            if periodo_analisis_ticks > 0:
                now_tick = cv2.getTickCount()
                if now_tick < next_due:
                    if args.display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                    time.sleep(0.001)
                    continue
                next_due = cv2.getTickCount() + periodo_analisis_ticks

            if grabber is not None:
                ok, frame = grabber.read_copy()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                print(
                    f"[RETRY] Sin frame USB cam {args.camera}. "
                    f"Reintentando en {REINTENTO_CAPTURA_SEG}s..."
                )
                if grabber is not None:
                    grabber.stop()
                    grabber = None
                cap.release()

                while True:
                    if args.display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            return
                    time.sleep(REINTENTO_CAPTURA_SEG)
                    cap = abrir_usb_con_calentamiento(args.camera)
                    if cap is not None:
                        if USAR_HILO_CAPTURA:
                            grabber = UltimoFrameCamara(cap)
                            grabber.start()
                            if esperar_primer_frame_grabber(grabber):
                                print("[RETRY] Camara USB reconectada. Captura en hilo reanudada.")
                                break
                            grabber.stop()
                            grabber = None
                            cap.release()
                            cap = None
                            print(
                                f"[RETRY] USB reabrio pero sin primer frame del hilo. "
                                f"Nuevo intento en {REINTENTO_CAPTURA_SEG}s..."
                            )
                        else:
                            print("[RETRY] Camara USB reconectada.")
                            break
                    print(
                        f"[RETRY] Fallo reapertura USB. Nuevo intento en {REINTENTO_CAPTURA_SEG}s..."
                    )
                continue

            frame_count += 1
            log_fps_analisis(frame_count, t0_tick, frame)

            img_height, img_width = frame.shape[:2]
            letterbox_img, lb_meta = letterbox_bgr(
                frame,
                (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
                RETINAFACE_LETTERBOX_FILL,
            )
            tensor = _preprocess_for_onnx(inp0, letterbox_img)
            ort_outputs = session.run(None, {input_name: tensor})

            dets = retinaface_dets_desde_rknn_outputs(
                list(ort_outputs),
                img_width=img_width,
                img_height=img_height,
                aspect_ratio=lb_meta.aspect_ratio,
                offset_x=lb_meta.offset_x,
                offset_y=lb_meta.offset_y,
                score_deteccion=RETINAFACE_SCORE_DETECCION,
                score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
            )

            n_faces = dets.shape[0]
            if n_faces > 0:
                msg_parts = [f"face({float(row[4]):.2f})" for row in dets]
                print("Detecciones: " + ", ".join(msg_parts))

            for data in dets:
                text = "{:.4f}".format(float(data[4]))
                di = list(map(int, data))
                cv2.rectangle(
                    frame, (di[0], di[1]), (di[2], di[3]), (0, 0, 255), 2
                )
                cx, cy = di[0], di[1] + 12
                cv2.putText(
                    frame,
                    text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                )
                cv2.circle(frame, (di[5], di[6]), 1, (0, 0, 255), 5)
                cv2.circle(frame, (di[7], di[8]), 1, (0, 255, 255), 5)
                cv2.circle(frame, (di[9], di[10]), 1, (255, 0, 255), 5)
                cv2.circle(frame, (di[11], di[12]), 1, (0, 255, 0), 5)
                cv2.circle(frame, (di[13], di[14]), 1, (255, 0, 0), 5)

            if n_faces > 0 and not args.no_save:
                now_save = cv2.getTickCount()
                if now_save - last_save_tick >= save_interval_ticks:
                    os.makedirs(FILE_DIR, exist_ok=True)
                    file_path = construir_file_path_dia()
                    if cv2.imwrite(file_path, frame):
                        print(f"[SAVE] {file_path}")
                        last_save_tick = now_save
                    else:
                        print(f"[SAVE] error al guardar: {file_path}")

            if args.display:
                cv2.imshow("retinaface onnx USB", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
