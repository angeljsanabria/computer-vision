"""
Alineacion facial estilo ArcFace para entrada MobileFaceNet (112x112).

Port de la logica de InsightFace ``face_align.py`` sin dependencia de
``insightface`` ni ``skimage`` en runtime.

Referencia upstream (copia en repo):
  inference/reference/insightface_face_align.py

  https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

Cambio respecto al original: ``estimate_norm`` usa ``cv2.estimateAffinePartial2D``
en lugar de ``skimage.transform.SimilarityTransform`` (misma similitud afín).
"""
from __future__ import annotations

import cv2
import numpy as np

# Plantilla ArcFace en canvas 112x112 (identica al upstream).
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

MOBILEFACENET_ALIGN_SIZE = 112


def landmarks_from_det_row(det: np.ndarray) -> np.ndarray:
    """
    Landmarks (5, 2) desde fila RetinaFace (15,).

    Orden: ojo_izq, ojo_der, nariz, boca_izq, boca_der (indices 5-14).
    """
    if det.shape[0] < 15:
        raise ValueError(f"fila det invalida: shape={det.shape}")
    return det[5:15].reshape(5, 2).astype(np.float32)


def estimate_norm(
    lmk: np.ndarray,
    image_size: int = 112,
    mode: str = "arcface",
) -> np.ndarray:
    """
    Matriz afin 2x3: landmarks fuente -> plantilla ``arcface_dst`` escalada.

    API compatible con InsightFace ``estimate_norm`` (solo mode='arcface').
    """
    if mode != "arcface":
        raise ValueError(f"mode no soportado: {mode!r}")
    if lmk.shape != (5, 2):
        raise ValueError(f"lmk debe ser (5, 2), got {lmk.shape}")
    if image_size % 112 != 0 and image_size % 128 != 0:
        raise ValueError("image_size debe ser multiplo de 112 o 128")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    src = lmk.astype(np.float32, copy=False)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    if matrix is None:
        raise RuntimeError("estimateAffinePartial2D fallo (landmarks degenerados?)")
    return matrix


def norm_crop(
    img: np.ndarray,
    landmark: np.ndarray,
    image_size: int = 112,
    mode: str = "arcface",
) -> np.ndarray:
    """Parche alineado image_size x image_size (misma API que InsightFace)."""
    matrix = estimate_norm(landmark, image_size, mode)
    return cv2.warpAffine(
        img,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
    )


def norm_crop2(
    img: np.ndarray,
    landmark: np.ndarray,
    image_size: int = 112,
    mode: str = "arcface",
) -> tuple[np.ndarray, np.ndarray]:
    """Como ``norm_crop`` pero devuelve tambien la matriz 2x3."""
    matrix = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(
        img,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
    )
    return warped, matrix


def align_face_from_det_row(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    image_size: int = MOBILEFACENET_ALIGN_SIZE,
) -> np.ndarray:
    """Atajo pipeline: frame BGR + fila RetinaFace -> cara alineada BGR."""
    lmk = landmarks_from_det_row(det_row)
    return norm_crop(frame_bgr, lmk, image_size=image_size)
