"""FaceMesh a partir de una fila RetinaFace (crop + infer + remap)."""
from __future__ import annotations

import numpy as np

from inference.face_crop import bbox_crop_with_margin
from inference.facemesh.postprocess import landmarks_mesh_to_frame
from inference.facemesh.preprocess import crop_to_bgr192
from inference.types import FaceMeshLandmarks


class FaceMeshEstimatorLike:
    """Contrato minimo para ``estimate_from_det`` (evita import circular)."""

    def estimate(self, face_bgr: np.ndarray) -> np.ndarray: ...


def estimate_from_det(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    estimator: FaceMeshEstimatorLike,
    *,
    margin_frac: float,
) -> FaceMeshLandmarks | None:
    """
    Recorta la cara, infiere FaceMesh y devuelve landmarks en coords del frame.

    Retorna ``None`` si el recorte queda vacio.
    """
    img_h, img_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_crop_with_margin(det_row, img_w, img_h, margin_frac)
    face_crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    if face_crop.size == 0:
        return None

    patch = crop_to_bgr192(face_crop)
    mesh_pts = estimator.estimate(patch)
    frame_pts = landmarks_mesh_to_frame(mesh_pts, (x1, y1, x2, y2))
    return FaceMeshLandmarks(
        points=frame_pts,
        crop_xyxy=(x1, y1, x2, y2),
    )
