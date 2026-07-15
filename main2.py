"""Prueba rapida PFLD 106 landmarks (lite.onnx) con webcam."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from configs import settings as s
from inference import build_face_detector
from inference.face_crop import bbox_crop_with_margin
from inference.retinaface.select_best import mejores_caras

_SRC_DIR = Path(__file__).resolve().parent
_PFLD_MODEL = _SRC_DIR / "models_onnx" / "lite.onnx"
_RETINAFACE_MODEL = _SRC_DIR / "models_onnx" / "RetinaFace_mobile320.onnx"
_PFLD_SIZE = 112

session = ort.InferenceSession(
    str(_PFLD_MODEL),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
# lite.onnx: output1 = feature map, output = landmarks [1, 212]
_LANDMARK_OUT_IDX = 1

face_detector = build_face_detector(
    "pc",
    str(_RETINAFACE_MODEL),
    s.RETINAFACE_SCORE_DETECCION,
    s.RETINAFACE_SCORE_PRE_NMS,
)
if face_detector is None:
    raise RuntimeError("No se pudo crear RetinaFace")

cap = cv2.VideoCapture(0)
cv2.namedWindow("PFLD 106 landmarks", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("PFLD 106 landmarks", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    detections = mejores_caras(face_detector.detect(frame), top_n=1)

    if detections.has_faces:
        det = detections.dets[0]
        x1, y1, x2, y2 = bbox_crop_with_margin(
            det,
            frame_w,
            frame_h,
            s.FACE_CROP_MARGIN_FRAC,
        )
        face_crop = frame[y1 : y2 + 1, x1 : x2 + 1]

        if face_crop.size != 0:
            # PFLD requiere BGR 112x112, NCHW, float32 normalizado a [0, 1].
            pfld_image = cv2.resize(
                face_crop,
                (_PFLD_SIZE, _PFLD_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            input_data = (
                pfld_image.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
                / 255.0
            )
            outputs = session.run(None, {input_name: input_data})
            landmarks = outputs[_LANDMARK_OUT_IDX].reshape(-1, 2)

            crop_w = x2 - x1 + 1
            crop_h = y2 - y1 + 1
            for x_n, y_n in landmarks:
                x = int(x1 + x_n * crop_w)
                y = int(y1 + y_n * crop_h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imshow("PFLD 106 landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
