"""Constantes MobileFaceNet (foamliu / export_models)."""
from __future__ import annotations

import numpy as np

INPUT_HEIGHT = 112
INPUT_WIDTH = 112
INPUT_HW = (INPUT_WIDTH, INPUT_HEIGHT)
EMBED_DIM = 128

# RGB [0, 1] -> (x - mean) / std (PC ONNX). RK3568: mean/std en .rknn.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
