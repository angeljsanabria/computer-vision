"""Constantes del modelo RetinaFace mobile 320 (Zoo Rockchip / ONNX)."""
from __future__ import annotations

import numpy as np

INPUT_WIDTH = 320
INPUT_HEIGHT = 320
INPUT_HW = (INPUT_HEIGHT, INPUT_WIDTH)

# Relleno letterbox del demo oficial (BGR, mismo valor en los 3 canales).
LETTERBOX_FILL = 114

# Media Caffe del entrenamiento original (preproceso ONNX en PC).
MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)

# Umbral IoU del NMS del demo Zoo (filtro geometrico, no score de cara).
NMS_IOU = 0.5

BOX_SCALE = np.array(
    [INPUT_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT],
    dtype=np.float64,
)
LANDMARK_SCALE = np.array(
    [INPUT_WIDTH, INPUT_HEIGHT] * 5,
    dtype=np.float64,
)
