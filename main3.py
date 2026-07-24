"""Prueba FaceMesh 468 landmarks sobre un crop detectado por RetinaFace."""
from __future__ import annotations

from pathlib import Path

import cv2

from configs import settings as s
from inference import build_face_detector, build_face_mesh
from inference.facemesh.from_detection import estimate_from_det
from inference.retinaface.select_best import mejores_caras
from ui.facemesh_overlay import draw_facemesh_landmarks

_SRC_DIR = Path(__file__).resolve().parent
_RETINAFACE_MODEL = _SRC_DIR / "models_onnx" / "RetinaFace_mobile320.onnx"
_FACE_MESH_MODEL = _SRC_DIR / "models_onnx" / "face_mesh_192x192.onnx"
_WINDOW = "FaceMesh 468 (192x192)"

face_detector = build_face_detector(
    "pc",
    str(_RETINAFACE_MODEL),
    s.RETINAFACE_SCORE_DETECCION,
    s.RETINAFACE_SCORE_PRE_NMS,
)
if face_detector is None:
    raise RuntimeError("No se pudo crear RetinaFace")

face_mesh = build_face_mesh("pc", str(_FACE_MESH_MODEL))
if face_mesh is None:
    raise RuntimeError("No se pudo crear FaceMesh")

cap = cv2.VideoCapture(0)
cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = None
        detections = mejores_caras(face_detector.detect(frame), top_n=1)
        if detections.has_faces:
            landmarks = estimate_from_det(
                frame,
                detections.dets[0],
                face_mesh,
                margin_frac=s.FACE_CROP_MARGIN_FRAC,
            )

        vis = draw_facemesh_landmarks(frame, landmarks, draw_crop_rect=True)
        cv2.imshow(_WINDOW, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    face_mesh.release()
    cap.release()
    cv2.destroyAllWindows()
