"""
RetinaFace (.rknn) en placa Rockchip: camara IP por RTSP + inferencia RKNNLite.

Postproceso en utils (letterbox + retinaface_dets_desde_rknn_outputs).

python3 RetinaFace_lite_from_ip_cam.py
python3 RetinaFace_lite_from_ip_cam.py --display
python3 RetinaFace_lite_from_ip_cam.py --no-save
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa (rknnlite). "
        "Ej.: pip3 install --user rknn_toolkit_lite2-....whl"
    ) from e


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


ROOT = _project_root_with_utils(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aux_tools_retinaface import (
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr

# --- RTSP: mismos campos que detect_yolov8_rknn_lite_cam_ip_person.py ---
USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RTSP_PORT_CAM = 554
RES_HIGH_CAM = "Preview_01_main"
RES_LOW_CAM = "Preview_01_sub"
STREAM_CAM = RES_LOW_CAM
RTSP_URL = f"rtsp://{USER_CAM}:{PASS_CAM}@{IP_CAM}:{RTSP_PORT_CAM}/{STREAM_CAM}"

USAR_HILO_CAPTURA = True
TAMANO_BUFFER_CAMARA = 1
MAX_FPS_ANALISIS = 2.0
LOG_CADA_CAPS = 10
TIME_SAVE_DETECTION = 3 * 60
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_retinaface_rtsp"
REINTENTO_CONEXION_SEG = 10

RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2


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


def abrir_rtsp_con_calentamiento(rtsp_url: str) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
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
    Evita que mientras la NPU infiere se acumulen frames viejos en el buffer.
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
    _wip = Path(__file__).resolve().parent
    _repo = _wip.parent
    default_rknn = _repo / "model" / "RetinaFace_mobile320_2.rknn"

    parser = argparse.ArgumentParser(
        description="RetinaFace RKNN desde camara IP RTSP (captura como YOLO cam_ip_person)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_rknn),
        help="Ruta al .rknn",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        default="",
        help="URL RTSP completa; si vacio se usa RTSP_URL del script.",
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
    args = parser.parse_args()

    rtsp_url = args.rtsp_url.strip() or RTSP_URL

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el modelo: {args.model_path}")

    cap = abrir_rtsp_con_calentamiento(rtsp_url)
    while cap is None:
        print(
            f"[RETRY] No se pudo abrir/calentar RTSP. Reintento en {REINTENTO_CONEXION_SEG}s..."
        )
        time.sleep(REINTENTO_CONEXION_SEG)
        cap = abrir_rtsp_con_calentamiento(rtsp_url)

    rknn = RKNNLite()
    print("--> load_rknn", args.model_path)
    if rknn.load_rknn(args.model_path) != 0:
        cap.release()
        raise SystemExit("load_rknn failed")

    print("--> init_runtime")
    if rknn.init_runtime() != 0:
        rknn.release()
        cap.release()
        raise SystemExit("init_runtime failed")

    grabber: UltimoFrameCamara | None = None
    if USAR_HILO_CAPTURA:
        grabber = UltimoFrameCamara(cap)
        grabber.start()
        if not esperar_primer_frame_grabber(grabber):
            grabber.stop()
            cap.release()
            rknn.release()
            raise SystemExit(
                "Camara conectada pero el hilo no recibio el primer frame. "
                "Revisar estabilidad de stream RTSP."
            )
        print("Captura en hilo auxiliar activa (menos latencia por buffer).")

    if args.display:
        print("Camara lista. Modo display (q o ESC para salir).")
    else:
        print("Camara lista. Sin display (Ctrl+C para salir).")

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
                    f"[RETRY] Sin frame de camara RTSP. Reintentando en {REINTENTO_CONEXION_SEG}s..."
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
                    time.sleep(REINTENTO_CONEXION_SEG)
                    cap = abrir_rtsp_con_calentamiento(rtsp_url)
                    if cap is not None:
                        if USAR_HILO_CAPTURA:
                            grabber = UltimoFrameCamara(cap)
                            grabber.start()
                            if esperar_primer_frame_grabber(grabber):
                                print("[RETRY] Camara reconectada. Captura en hilo reanudada.")
                                break
                            grabber.stop()
                            grabber = None
                            cap.release()
                            cap = None
                            print(
                                f"[RETRY] Reconecto RTSP pero sin primer frame del hilo. "
                                f"Nuevo intento en {REINTENTO_CONEXION_SEG}s..."
                            )
                        else:
                            print("[RETRY] Camara reconectada.")
                            break
                    print(
                        f"[RETRY] Fallo reconexion RTSP. Nuevo intento en {REINTENTO_CONEXION_SEG}s..."
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
            infer_rgb = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(infer_rgb, axis=0)

            outputs = rknn.inference(inputs=[input_tensor])
            if not outputs:
                if args.display:
                    cv2.imshow("retinaface rknn RTSP", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                continue

            dets = retinaface_dets_desde_rknn_outputs(
                outputs,
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
                cv2.imshow("retinaface rknn RTSP", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if grabber is not None:
            grabber.stop()
        rknn.release()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
