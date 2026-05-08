"""
Herramientas de postproceso SOLO validas para RetinaFace (Pytorch_Retinaface / rknn_model_zoo).

No aplicar a YOLO ni a otros detectores: las anclas, variances y formas de salida
estan acopladas a este modelo.

Constantes por defecto alineadas al ejemplo MobileNet 0.25 entrada 320x320 (Zoo Rockchip).
"""
from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any

import numpy as np

# --- Tamano de entrada del modelo (MobileNet 0.25 tipico en Zoo) ---
RETINAFACE_INPUT_HEIGHT = 320
RETINAFACE_INPUT_WIDTH = 320
RETINAFACE_INPUT_HW: tuple[int, int] = (RETINAFACE_INPUT_HEIGHT, RETINAFACE_INPUT_WIDTH)

# Valor de relleno letterbox en el demo oficial (BGR constante por canal).
RETINAFACE_LETTERBOX_FILL = 114

# Escalado de cajas y landmarks del espacio normalizado [0,1] al tamano de entrada del modelo.
RETINAFACE_BOX_SCALE = np.array(
    [
        RETINAFACE_INPUT_WIDTH,
        RETINAFACE_INPUT_HEIGHT,
        RETINAFACE_INPUT_WIDTH,
        RETINAFACE_INPUT_HEIGHT,
    ],
    dtype=np.float64,
)
RETINAFACE_LANDMARK_SCALE = np.array(
    [RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT] * 5,
    dtype=np.float64,
)


def prior_box(image_size: tuple[int, int], *, log_priors: bool = False) -> np.ndarray:
    """
    Genera priors (anclas) en coordenadas normalizadas para RetinaFace.

    image_size: (alto, ancho) del tensor de entrada, p. ej. (320, 320).
    log_priors: si True, imprime tamaño y num_priors (depuracion).

    Retorna array forma (N, 4) con [cx, cy, w, h] normalizados.
    """
    anchors: list[float] = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps
    ]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors, dtype=np.float64).reshape(-1, 4)
    if log_priors:
        print("image_size:", image_size, " num_priors=", output.shape[0])
    return output


def box_decode(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """
    Decodifica regresiones loc respecto a priors (solo RetinaFace / mismo encoding).

    loc: (num_priors, 4), priors: (num_priors, 4).
    Variances fijas [0.1, 0.2] del entrenamiento original.
    """
    variances = [0.1, 0.2]
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """
    Decodifica 5 landmarks x 2 coordenadas (10 valores) respecto a priors.
    Solo RetinaFace con este layout de salida.
    """
    variances = [0.1, 0.2]
    return np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    """
    NMS tipo RetinaFace demo: dets filas (x1, y1, x2, y2, score).
    Implementacion baseline con +1 en areas (igual al Zoo Python).

    Solo garantizada para este flujo de cajas; otros modelos pueden necesitar otra metrica.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def split_outputs(outputs: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa salidas RKNN/RKNNLite del RetinaFace_mobile320: loc, conf, landm.

    Si hay exactamente 3 tensores se respeta el orden; si no, se eligen por ultima dimension 4, 2, 10.
    """
    arrs = [np.array(o) for o in outputs]
    if len(arrs) == 3:
        return arrs[0], arrs[1], arrs[2]
    loc = next(t for t in arrs if t.shape[-1] == 4)
    conf = next(t for t in arrs if t.shape[-1] == 2)
    landm = next(t for t in arrs if t.shape[-1] == 10)
    return loc, conf, landm


def retinaface_dets_desde_rknn_outputs(
    outputs: list[Any],
    *,
    img_width: int,
    img_height: int,
    aspect_ratio: float,
    offset_x: int,
    offset_y: int,
    score_deteccion: float,
    score_pre_nms: float = 0.02,
    nms_iou: float = 0.5,
    log_priors: bool = False,
) -> np.ndarray:
    """
    Postproceso completo de salidas ``rknn.inference`` para RetinaFace (MobileNet 0.25 / Zoo).

    Convierte ``loc``, ``conf``, ``landm`` en filas listas para dibujar: cada fila es
    ``[x1, y1, x2, y2, score, 10 coords landmarks]`` en **pixeles de la imagen original**
    (tras des-letterbox con ``aspect_ratio`` y ``offset_*``).

    Args:
        outputs: Lista de tensores tal como devuelve ``RKNNLite.inference``.
        img_width, img_height: Tamano del frame **original** (sin letterbox).
        aspect_ratio, offset_x, offset_y: Metadatos del mismo letterbox usado antes de la red
            (mismo significado que en ``letterbox_bgr`` / demo Zoo).
        score_deteccion: Umbral **final** sobre ``conf[:, 1]`` (cara). Solo se devuelven
            detecciones con ``score >= score_deteccion`` (lo define el script llamador).
        score_pre_nms: Filtro debil previo al NMS (tipico 0.02 en el demo Zoo).
        nms_iou: Umbral de solapamiento para ``nms`` (no es score de cara).
        log_priors: Si True, ``prior_box`` imprime depuracion (por defecto silencioso).

    Returns:
        ``ndarray`` forma ``(N, 15)``, ``float32``. Si no hay detecciones sobre el umbral,
        ``N == 0``. Orden: score descendente.
    """
    loc, conf, landmarks = split_outputs(outputs)
    priors = prior_box(
        (RETINAFACE_INPUT_HEIGHT, RETINAFACE_INPUT_WIDTH),
        log_priors=log_priors,
    )
    boxes = box_decode(loc.squeeze(0), priors)
    boxes = boxes * RETINAFACE_BOX_SCALE // 1
    boxes[..., 0::2] = np.clip(
        (boxes[..., 0::2] - offset_x) / aspect_ratio,
        0,
        img_width,
    )
    boxes[..., 1::2] = np.clip(
        (boxes[..., 1::2] - offset_y) / aspect_ratio,
        0,
        img_height,
    )
    scores = conf.squeeze(0)[:, 1]
    landmarks = decode_landm(landmarks.squeeze(0), priors)
    landmarks = landmarks * RETINAFACE_LANDMARK_SCALE // 1
    landmarks[..., 0::2] = np.clip(
        (landmarks[..., 0::2] - offset_x) / aspect_ratio,
        0,
        img_width,
    )
    landmarks[..., 1::2] = np.clip(
        (landmarks[..., 1::2] - offset_y) / aspect_ratio,
        0,
        img_height,
    )

    inds = np.where(scores > score_pre_nms)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, nms_iou)
    dets = dets[keep, :]
    landmarks = landmarks[keep]
    dets = np.concatenate((dets, landmarks), axis=1)

    mask = dets[:, 4] >= score_deteccion
    dets = dets[mask]
    return dets
