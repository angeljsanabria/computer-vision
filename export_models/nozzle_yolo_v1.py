"""
Fuel nozzle: YOLOv8 ONNX en PC con camara USB o IP RTSP.

Basado en use_model_yolov8/detect_yolov8_rknn_lite_cam_ip_person.py
(hilo de captura, reconexion RTSP, limite de FPS de analisis) adaptado a
Ultralytics + ONNX en PC (sin RKNN).

Modelo: Yolo-Weights/yolov8n_nozzle.onnx (fine-tune fuel nozzle, opset 19).

Uso (desde la raiz del repo):
  python export_models/nozzle_yolo_v1.py --modo usb --display
  python export_models/nozzle_yolo_v1.py --modo rtsp --display
  python export_models/nozzle_yolo_v1.py --modo rtsp --rtsp-url "rtsp://user:pass@ip:554/Preview_01_sub"

Salir: q o ESC en la ventana (o Ctrl+C sin display).
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import settings as s
from utils.camera_opencv import abrir_camara, preparar_camara

ONNX_PATH = ROOT / "Yolo-Weights" / "yolov8n_nozzle.onnx"

CONF_MIN = 0.35
USAR_HILO_CAPTURA = True
TAMANO_BUFFER_CAMARA = int(s.BUFFER_SIZE)
MAX_FPS_ANALISIS = 0.0  # 0 = sin limite en PC; subir si la CPU no alcanza
LOG_CADA_CAPS = int(s.LOG_CADA_N_FRAMES)
TIME_SAVE_DETECTION = 3 * 60
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_nozzle_onnx"
REINTENTO_CONEXION_SEG = float(s.REINTENTO_SEG)
CALENTAMIENTO_RTSP = int(s.WARMUP_FRAMES)


def configurar_buffer_camara(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, TAMANO_BUFFER_CAMARA)
    except Exception:
        pass


def log_fps_analisis(frame_count: int, t0_tick: int, frame) -> None:
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

    for _ in range(CALENTAMIENTO_RTSP):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            configurar_buffer_camara(cap)
            return cap

    cap.release()
    return None


def abrir_usb_con_calentamiento(indice: int) -> cv2.VideoCapture | None:
    cap = abrir_camara(indice)
    if cap is None:
        return None
    if not preparar_camara(cap):
        cap.release()
        return None
    configurar_buffer_camara(cap)
    return cap


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
    """Un hilo lee cap.read(); el bucle principal usa el ultimo frame (menos latencia)."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
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

    def read_copy(self) -> tuple[bool, object]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def _etiquetas_deteccion(result) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    out: list[str] = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, str(cls_id))
        out.append(f"{label}({conf:.2f})")
    return out


def _abrir_rtsp_con_reintentos(rtsp_url: str) -> cv2.VideoCapture:
    cap = abrir_rtsp_con_calentamiento(rtsp_url)
    while cap is None:
        print(
            f"[RETRY] No se pudo abrir/calentar RTSP. Reintento en {REINTENTO_CONEXION_SEG}s..."
        )
        time.sleep(REINTENTO_CONEXION_SEG)
        cap = abrir_rtsp_con_calentamiento(rtsp_url)
    return cap


def _iniciar_grabber(cap: cv2.VideoCapture) -> UltimoFrameCamara | None:
    if not USAR_HILO_CAPTURA:
        return None
    grabber = UltimoFrameCamara(cap)
    grabber.start()
    if not esperar_primer_frame_grabber(grabber):
        grabber.stop()
        raise SystemExit(
            "Camara conectada pero el hilo no recibio el primer frame."
        )
    print("Captura en hilo auxiliar activa (menos latencia por buffer).")
    return grabber


def _reconectar_rtsp(
    rtsp_url: str,
    grabber: UltimoFrameCamara | None,
    cap: cv2.VideoCapture | None,
    display: bool,
) -> tuple[cv2.VideoCapture, UltimoFrameCamara | None]:
    if grabber is not None:
        grabber.stop()
    if cap is not None:
        cap.release()

    while True:
        if display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                raise SystemExit(0)
        time.sleep(REINTENTO_CONEXION_SEG)
        cap = abrir_rtsp_con_calentamiento(rtsp_url)
        if cap is None:
            print(
                f"[RETRY] Fallo reconexion RTSP. Nuevo intento en {REINTENTO_CONEXION_SEG}s..."
            )
            continue
        grabber = _iniciar_grabber(cap)
        print("[RETRY] Camara RTSP reconectada.")
        return cap, grabber


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deteccion fuel nozzle (YOLOv8 ONNX) desde USB o RTSP."
    )
    parser.add_argument(
        "--modo",
        choices=("usb", "rtsp"),
        default=s.MODO.lower() if s.MODO.lower() in ("usb", "rtsp") else "usb",
        help="Fuente de video: webcam USB o camara IP RTSP.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Mostrar ventana OpenCV (por defecto activo si DISPLAY_IS_ENABLE en settings).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="No abrir ventana (headless).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar capturas periodicas con deteccion.",
    )
    parser.add_argument("--conf", type=float, default=CONF_MIN, help="Umbral de confianza.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ONNX_PATH,
        help="Ruta al modelo ONNX nozzle.",
    )
    parser.add_argument(
        "--usb-index",
        type=int,
        default=int(s.USB_INDEX),
        help="Indice de camara USB (0 en PC; en RK3568 suele ser 10 u 11).",
    )
    parser.add_argument(
        "--rtsp-url",
        default="",
        help="URL RTSP completa; si vacio usa IP_CAM_RTSP_URL de settings.",
    )
    args = parser.parse_args()

    display = s.DISPLAY_IS_ENABLE if not args.no_display else False
    if args.display:
        display = True

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(
            f"No se encuentra el modelo: {onnx_path}\n"
            "Exporta antes: python yolo_train/export_nozzle_onnx.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Instala ultralytics: pip install ultralytics") from e

    model = YOLO(str(onnx_path))
    print(f"Modelo:  {onnx_path.name}")
    print(f"Clases:  {model.names}")
    print(f"Modo:    {args.modo}")
    print(f"Conf:    {args.conf}")

    rtsp_url = args.rtsp_url.strip() or s.IP_CAM_RTSP_URL
    grabber: UltimoFrameCamara | None = None
    cap: cv2.VideoCapture | None = None

    if args.modo == "usb":
        cap = abrir_usb_con_calentamiento(args.usb_index)
        if cap is None:
            raise SystemExit(
                f"No se pudo abrir la camara USB index {args.usb_index}."
            )
        grabber = _iniciar_grabber(cap)
    else:
        print(f"RTSP: {rtsp_url}")
        cap = _abrir_rtsp_con_reintentos(rtsp_url)
        grabber = _iniciar_grabber(cap)

    if display:
        print("Listo. q o ESC para salir.")
    else:
        print("Listo. Modo sin display (Ctrl+C para salir).")

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
    win = "fuel nozzle ONNX"

    try:
        while True:
            if periodo_analisis_ticks > 0:
                now_tick = cv2.getTickCount()
                if now_tick < next_due:
                    if display:
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
                if args.modo == "rtsp":
                    print(
                        f"[RETRY] Sin frame RTSP. Reintentando en {REINTENTO_CONEXION_SEG}s..."
                    )
                    cap, grabber = _reconectar_rtsp(rtsp_url, grabber, cap, display)
                    continue
                time.sleep(0.01)
                continue

            frame_count += 1
            log_fps_analisis(frame_count, t0_tick, frame)

            results = model(frame, conf=args.conf, verbose=False)
            result = results[0]
            labels = _etiquetas_deteccion(result)
            if labels:
                print("Detecciones: " + ", ".join(labels))

            vis = result.plot()

            if labels and not args.no_save:
                now_tick_save = cv2.getTickCount()
                if (now_tick_save - last_save_tick) >= save_interval_ticks:
                    os.makedirs(FILE_DIR, exist_ok=True)
                    file_path = construir_file_path_dia()
                    if cv2.imwrite(file_path, vis):
                        print(f"[SAVE] deteccion guardada: {file_path}")
                        last_save_tick = now_tick_save
                    else:
                        print(f"[SAVE] error al guardar: {file_path}")

            if display:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if grabber is not None:
            grabber.stop()
        if cap is not None:
            cap.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
