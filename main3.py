"""Prueba FaceMesh 468 landmarks sobre un crop detectado por RetinaFace."""
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
_FACE_MESH_MODEL = _SRC_DIR / "models_onnx" / "face_mesh_192x192.onnx"
_RETINAFACE_MODEL = _SRC_DIR / "models_onnx" / "RetinaFace_mobile320.onnx"
_FACE_MESH_SIZE = 192

session = ort.InferenceSession(
    str(_FACE_MESH_MODEL),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

face_detector = build_face_detector(
    "pc",
    str(_RETINAFACE_MODEL),
    s.RETINAFACE_SCORE_DETECCION,
    s.RETINAFACE_SCORE_PRE_NMS,
)
if face_detector is None:
    raise RuntimeError("No se pudo crear RetinaFace")

cap = cv2.VideoCapture(0)
cv2.namedWindow("MediaPipe Face Mesh (192x192)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("MediaPipe Face Mesh (192x192)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            # FaceMesh requiere RGB 192x192, NCHW y float32 en [0, 1].
            mesh_image = cv2.resize(
                face_crop,
                (_FACE_MESH_SIZE, _FACE_MESH_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_BGR2RGB)
            input_data = (
                mesh_image.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
                / 255.0
            )

            outputs = session.run(None, {input_name: input_data})
            landmarks = outputs[0].reshape(-1, 3)

            crop_w = x2 - x1 + 1
            crop_h = y2 - y1 + 1
            for x_mesh, y_mesh, _ in landmarks:
                x = int(x1 + (x_mesh / _FACE_MESH_SIZE) * crop_w)
                y = int(y1 + (y_mesh / _FACE_MESH_SIZE) * crop_h)
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imshow("MediaPipe Face Mesh (192x192)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()