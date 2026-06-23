"""Normalizacion L2 de vectores de embedding."""
from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    flat = vec.reshape(-1).astype(np.float32, copy=False)
    n = float(np.linalg.norm(flat, ord=2))
    if n < 1e-12:
        return flat
    return (flat / n).astype(np.float32)
