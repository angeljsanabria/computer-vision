"""MobileFaceNet en placa RK3568 via RKNN Lite."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.mobilefacenet.constants import EMBED_DIM
from inference.mobilefacenet.norm import l2_normalize
from inference.mobilefacenet.preprocess import bgr112_to_rknn_nhwc

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # type: ignore[assignment,misc]


class MobileFaceNetEmbedderRk3568:
    """
    Embedding facial MobileFaceNet (RKNN) para RK3568.

    Entrada RGB uint8 112x112; mean/std ImageNet en el .rknn al export.
    """

    def __init__(self, model_path: str | Path) -> None:
        if RKNNLite is None:
            raise RuntimeError(
                "rknnlite no instalado. Instala RKNN-Toolkit-Lite2 en la placa "
                "(aarch64), p. ej. rknn_toolkit_lite2-2.3.2-...whl"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe modelo MobileFaceNet RKNN: {path}"
            )

        self._rknn: RKNNLite | None = RKNNLite()
        if self._rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"load_rknn failed: {path}")
        if self._rknn.init_runtime() != 0:
            self._rknn.release()
            raise RuntimeError(f"init_runtime failed: {path}")

        logging.info("MobileFaceNet RK3568 (RKNN) cargado: %s", path)

    @classmethod
    def from_settings(cls) -> MobileFaceNetEmbedderRk3568:
        from configs import settings as s

        return cls(model_path=s.mobilefacenet_model_rk3568_path())

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Parche BGR 112x112 -> vector (EMBED_DIM,) float32 L2-normalizado."""
        if self._rknn is None:
            raise RuntimeError("MobileFaceNet RK3568: runtime ya liberado")

        feed = bgr112_to_rknn_nhwc(face_bgr)
        outputs = self._rknn.inference(inputs=[feed])
        if not outputs:
            raise RuntimeError("MobileFaceNet RK3568: inference sin salida")

        vec = l2_normalize(np.asarray(outputs[0], dtype=np.float32))
        if vec.size != EMBED_DIM:
            raise RuntimeError(
                f"MobileFaceNet RKNN devolvio {vec.size} valores, esperado {EMBED_DIM}"
            )
        return vec
