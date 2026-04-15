"""
YOLOv8n (.rknn) en placa Rockchip: camara USB, deteccion COCO completa (80 clases).

Misma apertura que 2_ocv_cam.py / utils.camera_opencv: V4L2 en Linux y calentamiento.
En RK3568 la webcam suele ser indice 10 u 11.

Uso en RK3568:
  python3 export_models/detect_yolov8_rknn_lite_cam_person.py

Salir: tecla q o ESC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

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
OBJ_THRESH = 0.5
NMS_THRESH = 0.45

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


def main() -> None:
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

    print("Camara lista. q o ESC para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            fh, fw = frame.shape[:2]
            small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb, 0)

            outputs = rknn.inference(inputs=[inp])
            if not outputs:
                continue
            pred = np.array(outputs[0])

            boxes, scores, class_ids = postprocess_yolov8_ultralytics(
                pred, OBJ_THRESH, NMS_THRESH
            )
            if boxes is not None:
                boxes = scale_boxes_to_frame(boxes, fw, fh)
                for box, sc, cid in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    x1 = max(0, min(x1, fw - 1))
                    x2 = max(0, min(x2, fw - 1))
                    y1 = max(0, min(y1, fh - 1))
                    y2 = max(0, min(y2, fh - 1))
                    cid_i = int(cid)
                    label = CLASSES[cid_i] if cid_i < len(CLASSES) else str(cid_i)
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

            cv2.imshow("yolov8 rknn coco", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        rknn.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
