"""RetinaFace en placa RK3568 via RKNN Lite."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.types import FaceDetections


class RetinaFaceDetectorRk3568:
    """Detector facial RetinaFace (RKNN) para RK3568."""

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        score_pre_nms: float,
    ) -> None:
        self._model_path = Path(model_path)
        self._score_deteccion = float(score_deteccion)
        self._score_pre_nms = float(score_pre_nms)
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"No existe modelo RetinaFace RKNN: {self._model_path}"
            )
        # TODO: completar con toolkit de Rockchip (rknnlite, init, inference).
        logging.warning(
            "RetinaFace RK3568: stub sin inferencia real (%s)",
            self._model_path,
        )

    @classmethod
    def from_settings(cls) -> RetinaFaceDetectorRk3568:
        from configs import settings as s

        return cls(
            model_path=s.retinaface_model_rk3568_path(),
            score_deteccion=s.RETINAFACE_SCORE_DETECCION,
            score_pre_nms=s.RETINAFACE_SCORE_PRE_NMS,
        )

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        del frame_bgr
        # TODO: letterbox + bgr_to_rgb + rknn.inference + dets_desde_salidas_modelo
        return FaceDetections.empty()
