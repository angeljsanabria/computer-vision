"""
YOLOv8n (.rknn) en placa Rockchip usando snapshots HTTP (cmd=Snap) de camara IP.

Features adicionales:
- --display habilita la muestra en una ventana de la imagen procesada.
- Guardado de ultima captura del dia cuando hay deteccion valida, respetando el
  tiempo configurado en TIME_SAVE_DETECTION.
- --no-save impide que se guarde una imagen por dia.

Uso en RK3568:
  python3 export_models/model_api_snap.py
  python3 export_models/model_api_snap.py --display
  python3 export_models/model_api_snap.py --no-save
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa: pip3 install "
        "rknn-toolkit-lite/rknn_toolkit_lite2-2.3.2-cp310-....whl"
    ) from e

RKNN_PATH = ROOT / "models" / "yolov8n.rknn"

# Igual esquema que 1_ocv_img_ip_cam_just_save.py
USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RES_HIGH_CAM = "width=2560&height=1920"
RES_LOW_CAM = "width=640&height=480"
SNAP_HTTP_URL_RES_FULL = (
    f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={USER_CAM}&password={PASS_CAM}&{RES_HIGH_CAM}"
)
SNAP_HTTP_URL_RES_LOW = (
    f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={USER_CAM}&password={PASS_CAM}&{RES_LOW_CAM}"
)
SNAP_HTTP_URL = SNAP_HTTP_URL_RES_LOW

INPUT_SIZE = 640
OBJ_THRESH = 0.65
NMS_THRESH = 0.55
MAX_FPS_ANALISIS = 2.0
LOG_CADA_CAPS = 10
HTTP_TIMEOUT_S = 10
TIME_SAVE_DETECTION = 3 * 60  # segundos (3 minutos)
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_detection_snap"

CLASSES = (
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
    sx = frame_w / float(INPUT_SIZE)
    sy = frame_h / float(INPUT_SIZE)
    out = xyxy.copy()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    return out


def log_fps_analisis(frame_count: int, t0_tick: int, frame: np.ndarray) -> None:
    if frame_count % LOG_CADA_CAPS != 0:
        return
    ticks = cv2.getTickCount() - t0_tick
    dt = ticks / cv2.getTickFrequency()
    fps = frame_count / dt if dt > 0 else 0.0
    h, w = frame.shape[:2]
    print(f"[LOG] frame={frame_count} size={w}x{h} fps_aprox={fps:.2f}")


def obtener_frame_snap(url: str) -> np.ndarray | None:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_S, verify=False)
    except requests.exceptions.RequestException as e:
        print(f"Error de conexion Snap: {e}")
        return None

    if response.status_code != 200:
        print(f"Error HTTP Snap: {response.status_code}")
        return None

    # Convertimos los bytes recibidos directamente en un array de numpy
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    # Decodificamos el array a un formato que OpenCV entiende (BGR)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return frame


def construir_file_path_dia() -> str:
    dd_mm = time.strftime("%d_%m")
    file_name_img = f"{FILE_BASE_NAME_IMG}_{dd_mm}.jpg"
    return os.path.join(FILE_DIR, file_name_img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv8 RKNN desde Snap API (HTTP)."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Abrir ventana de OpenCV (por defecto no muestra).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar imagenes de deteccion.",
    )
    args = parser.parse_args()

    if not RKNN_PATH.is_file():
        raise SystemExit(f"No existe el modelo: {RKNN_PATH}")

    rknn = RKNNLite()
    print("--> load_rknn", RKNN_PATH)
    if rknn.load_rknn(str(RKNN_PATH)) != 0:
        raise SystemExit("load_rknn failed")

    print("--> init_runtime")
    if rknn.init_runtime() != 0:
        rknn.release()
        raise SystemExit("init_runtime failed")

    if args.display:
        print("Snap API lista. Modo display activo (q o ESC para salir).")
    else:
        print("Snap API lista. Modo sin display activo (salir con Ctrl+C).")

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

            frame = obtener_frame_snap(SNAP_HTTP_URL)
            if frame is None:
                continue

            frame_count += 1
            log_fps_analisis(frame_count, t0_tick, frame)

            fh, fw = frame.shape[:2]
            small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb, 0)

            outputs = rknn.inference(inputs=[inp])
            if outputs:
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

                    # Guardar evidencia solo si hubo deteccion valida y paso el tiempo minimo.
                    if not args.no_save:
                        now_tick_save = cv2.getTickCount()
                        dt_save_ticks = now_tick_save - last_save_tick
                        if dt_save_ticks >= save_interval_ticks:
                            os.makedirs(FILE_DIR, exist_ok=True)
                            file_path = construir_file_path_dia()
                            if cv2.imwrite(file_path, frame):
                                print(f"[SAVE] deteccion guardada: {file_path}")
                                last_save_tick = now_tick_save
                            else:
                                print(f"[SAVE] error al guardar: {file_path}")

            if args.display:
                cv2.imshow("yolov8 rknn snap api", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        rknn.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
