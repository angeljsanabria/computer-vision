"""
Inferencia YOLOv8 (.rknn) en placa Rockchip con RKNN-Toolkit-Lite2.

Uso en RK3568 (Python 3.10, wheel aarch64 2.3.2):
  python3 export_models/detect_yolov8_rknn_lite_img.py

Ajusta RKNN_PATH e IMG si hace falta. El modelo debe ser el exportado desde
Ultralytics ONNX con salida tipica (1, 84, 8400), mismo preproceso que en
exp_yolov8n_rknn.py (mean 0, std 255 -> entrada uint8 NHWC).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa: pip3 install --user "
        "rknn-toolkit-lite/rknn_toolkit_lite2-2.3.2-cp310-....whl"
    ) from e

ROOT = Path(__file__).resolve().parent.parent

# En la placa: misma ruta relativa al repo o absoluta
RKNN_PATH = ROOT / "rknn-toolkit-lite" / "yolov8n.rknn"
IMG_PATH = ROOT / "images" / "lily2.jpg"

INPUT_SIZE = 640
OBJ_THRESH = 0.25
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def postprocess_yolov8_ultralytics(pred: np.ndarray, conf_thres: float, iou_thres: float):
    """
    pred: (1, 4+nc, 8400) salida tipica YOLOv8n COCO desde Ultralytics -> RKNN.
    """
    if pred.ndim == 3:
        pred = pred[0]
    # (84, 8400) -> (8400, 84)
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


def main() -> None:
    if not RKNN_PATH.is_file():
        raise SystemExit(f"No existe el modelo: {RKNN_PATH}")
    if not IMG_PATH.is_file():
        raise SystemExit(f"No existe la imagen: {IMG_PATH}")

    rknn = RKNNLite()
    print("--> load_rknn", RKNN_PATH)
    if rknn.load_rknn(str(RKNN_PATH)) != 0:
        raise SystemExit("load_rknn failed")

    print("--> init_runtime")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime failed")

    bgr = cv2.imread(str(IMG_PATH))
    if bgr is None:
        raise SystemExit("No se pudo leer la imagen")
    img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img_rgb, 0)

    print("--> inference")
    outputs = rknn.inference(inputs=[inp])
    rknn.release()

    if not outputs:
        raise SystemExit("Sin salidas")
    pred = np.array(outputs[0])
    print("Salida shape:", pred.shape)

    boxes, scores, classes = postprocess_yolov8_ultralytics(
        pred, OBJ_THRESH, NMS_THRESH
    )
    if boxes is None:
        print("Sin detecciones (revisa umbral o formato de salida).")
        return

    for box, sc, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label = CLASSES[int(cl)] if int(cl) < len(CLASSES) else str(int(cl))
        print(f"{label} {sc:.3f} [{x1}, {y1}, {x2}, {y2}]")
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"{label} {sc:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    out_path = ROOT / "images" / "lily2_rknn_result.jpg"
    cv2.imwrite(str(out_path), bgr)
    print("Guardado:", out_path)


if __name__ == "__main__":
    main()
