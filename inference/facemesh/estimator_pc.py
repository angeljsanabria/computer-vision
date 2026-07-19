"""FaceMesh 468 landmarks en PC via ONNX Runtime."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.facemesh.constants import LANDMARK_DIM, NUM_LANDMARKS
from inference.facemesh.preprocess import bgr192_to_onnx_nchw

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment,misc]


class FaceMeshEstimatorPc:
    """468 landmarks faciales (ONNX 192x192) para desarrollo en PC."""

    def __init__(self, model_path: str | Path) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime no instalado. pip install onnxruntime"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo FaceMesh ONNX: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logging.debug("FaceMesh PC (ONNX) cargado: %s", path)

    def estimate(self, face_bgr: np.ndarray) -> np.ndarray:
        """Parche BGR 192x192 -> (468, 3) float32 en espacio mesh."""
        feed = bgr192_to_onnx_nchw(face_bgr)
        outputs = self._session.run(None, {self._input_name: feed})
        pts = np.asarray(outputs[0], dtype=np.float32).reshape(-1, LANDMARK_DIM)
        if pts.shape[0] != NUM_LANDMARKS:
            raise RuntimeError(
                f"FaceMesh ONNX devolvio {pts.shape[0]} puntos, esperado {NUM_LANDMARKS}"
            )
        return pts

    def release(self) -> None:
        """No-op (simetria con futuro RK3568)."""
        return None
