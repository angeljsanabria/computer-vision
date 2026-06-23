"""MobileFaceNet en PC via ONNX Runtime."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.mobilefacenet.constants import EMBED_DIM
from inference.mobilefacenet.norm import l2_normalize
from inference.mobilefacenet.preprocess import bgr112_to_onnx_nchw

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment,misc]


class MobileFaceNetEmbedderPc:
    """Embedding facial MobileFaceNet (ONNX) para desarrollo en PC."""

    def __init__(self, model_path: str | Path) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime no instalado. pip install onnxruntime"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo MobileFaceNet ONNX: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logging.info("MobileFaceNet PC (ONNX) cargado: %s", path)

    @classmethod
    def from_settings(cls) -> MobileFaceNetEmbedderPc:
        from configs import settings as s

        return cls(model_path=s.mobilefacenet_model_pc_path())

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Parche BGR 112x112 -> vector (EMBED_DIM,) float32 L2-normalizado."""
        feed = bgr112_to_onnx_nchw(face_bgr)
        out = self._session.run(
            [self._output_name], {self._input_name: feed}
        )[0]
        vec = l2_normalize(np.asarray(out, dtype=np.float32))
        if vec.size != EMBED_DIM:
            raise RuntimeError(
                f"MobileFaceNet ONNX devolvio {vec.size} valores, esperado {EMBED_DIM}"
            )
        return vec

    def release(self) -> None:
        """No-op (simetria con RK3568)."""
        return None
