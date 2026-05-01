"""
YOLOv8n (.rknn) en placa Rockchip: camara USB, deteccion COCO completa (80 clases).

Misma apertura que 2_ocv_cam.py / utils.camera_opencv: V4L2 en Linux y calentamiento.
En RK3568 la webcam suele ser indice 10 u 11.

Uso en RK3568:
  python3 export_models/detect_yolov8_rknn_lite_cam_person.py
  python3 export_models/detect_yolov8_rknn_lite_cam_person.py --no-display

Salir: tecla q o ESC.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.camera_opencv import abrir_camara, preparar_camara

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa: pip3 install --user "
        "rknn-toolkit-lite/rknn_toolkit_lite2-2.3.2-cp310-....whl"
    ) from e

RKNN_PATH = ROOT / "rknn-toolkit-lite" / "yolov8n.rknn"
# RK3568 + USB UVC: suele ser 10 u 11 (no 0)
CAMERA_INDEX = 10

INPUT_SIZE = 640
OBJ_THRESH = 0.65
NMS_THRESH = 0.55
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 9600

# Hilo de captura: la camara se lee al ritmo del driver; la inferencia RKNN no bloquea read().
# Asi se reduce la latencia y el efecto "video a camara lenta" por buffer lleno.
# Nota: si la NPU es mas lenta que la FPS de la camara, igual solo se etiquetan
# algunos frames por segundo; la ventana se refresca a esa cadencia.
USAR_HILO_CAPTURA = True
# En muchos backends V4L2 reduce cola interna (1 frame); si no soporta, OpenCV lo ignora.
TAMANO_BUFFER_CAMARA = 1

CLASSES_COMPLETE = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

CLASSES = (
    "person", "orange", "cell phone", "book", "chair", "keyboard",
)
# Paleta BGR para distinguir clases en pantalla
_COLORS_BGR = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (203, 192, 255),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def postprocess_yolov8_ultralytics(pred: np.ndarray, conf_thres: float, iou_thres: float):
    """
    pred: (1, 4+nc, 8400) salida tipica YOLOv8n COCO desde Ultralytics -> RKNN.
    Todas las clases; score = max prob por ancla.
    """
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T
    boxes_xywh = pred[:, :4]
    cls_logits = pred[:, 4:]
    cls_prob = sigmoid(cls_logits)
    scores = np.max(cls_prob, axis=1)
    class_ids = np.argmax(cls_prob, axis=1)

    mask = scores >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if len(scores) == 0:
        return None, None, None

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = []
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        union = area_i + area_r - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = rest[inds]

    keep = np.array(keep, dtype=np.int64)
    return xyxy[keep], scores[keep], class_ids[keep]


def scale_boxes_to_frame(xyxy: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """De espacio 640x640 (letterbox implicito: resize directo) a tamano del frame."""
    sx = frame_w / float(INPUT_SIZE)
    sy = frame_h / float(INPUT_SIZE)
    out = xyxy.copy()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    return out


def configurar_buffer_camara(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, TAMANO_BUFFER_CAMARA)
    except Exception:
        pass


def enviar_serial(ser: serial.Serial | None, mensaje: str) -> None:
    if ser is None:
        return
    try:
        ser.write((mensaje + "\n").encode())
    except Exception as e:
        print(f"Error al enviar por serial: {e}")


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
                time.sleep(0.001)

    def read_copy(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv8 RKNN desde camara USB."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="No abrir ventana de OpenCV (modo headless/sin monitor).",
    )
    args = parser.parse_args()
    ser: serial.Serial | None = None

    if not RKNN_PATH.is_file():
        raise SystemExit(f"No existe el modelo: {RKNN_PATH}")

    cap = abrir_camara(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(
            f"No se pudo abrir la camara index {CAMERA_INDEX} "
            "(prueba otro indice; en RK3568 USB suele ser 10 u 11)."
        )
    if not preparar_camara(cap):
        cap.release()
        raise SystemExit(
            "La camara abrio pero no entrego frames; prueba otro CAMERA_INDEX."
        )

    configurar_buffer_camara(cap)

    rknn = RKNNLite()
    print("--> load_rknn", RKNN_PATH)
    if rknn.load_rknn(str(RKNN_PATH)) != 0:
        cap.release()
        raise SystemExit("load_rknn failed")

    print("--> init_runtime")
    if rknn.init_runtime() != 0:
        rknn.release()
        cap.release()
        raise SystemExit("init_runtime failed")

    if serial is None:
        print("pyserial no esta instalado. Se desactiva envio serial.")
    else:
        try:
            ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                timeout=0.05,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] Serial conectado en "
                f"{SERIAL_PORT} @ {SERIAL_BAUDRATE} baudios."
            )
        except serial.SerialException as e:
            print(f"Error al conectar serial: {e}")
            ser = None

    grabber: UltimoFrameCamara | None = None
    if USAR_HILO_CAPTURA:
        grabber = UltimoFrameCamara(cap)
        grabber.start()
        print("Captura en hilo auxiliar activa (menos latencia por buffer).")

    if args.no_display:
        print("Camara lista. Modo sin display activo (salir con Ctrl+C).")
    else:
        print("Camara lista. q o ESC para salir.")

    try:
        while True:
            if grabber is not None:
                ok, frame = grabber.read_copy()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                if grabber is not None:
                    time.sleep(0.001)
                    if not args.no_display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                    continue
                break

            fh, fw = frame.shape[:2]
            small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb, 0)

            outputs = rknn.inference(inputs=[inp])
            if not outputs:
                if not args.no_display:
                    cv2.imshow("yolov8 rknn coco", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                continue
            pred = np.array(outputs[0])

            boxes, scores, class_ids = postprocess_yolov8_ultralytics(
                pred, OBJ_THRESH, NMS_THRESH
            )
            if boxes is not None:
                boxes = scale_boxes_to_frame(boxes, fw, fh)
                detected_labels: list[str] = []
                for box, sc, cid in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    x1 = max(0, min(x1, fw - 1))
                    x2 = max(0, min(x2, fw - 1))
                    y1 = max(0, min(y1, fh - 1))
                    y2 = max(0, min(y2, fh - 1))
                    cid_i = int(cid)
                    label = CLASSES[cid_i] if cid_i < len(CLASSES) else str(cid_i)
                    detected_labels.append(label)
                    color = _COLORS_BGR[cid_i % len(_COLORS_BGR)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {sc:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                detecciones_msg = "Detecciones: " + ", ".join(detected_labels)
                print(detecciones_msg)
                enviar_serial(ser, detecciones_msg)

            if not args.no_display:
                cv2.imshow("yolov8 rknn coco", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if grabber is not None:
            grabber.stop()
        rknn.release()
        cap.release()
        if ser is not None:
            ser.close()
            print("Puerto serial cerrado.")
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
