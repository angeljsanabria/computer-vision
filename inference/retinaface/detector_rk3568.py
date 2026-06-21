"""RetinaFace en placa RK3568 via RKNN Lite."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from inference.retinaface.constants import INPUT_HEIGHT, INPUT_WIDTH, LETTERBOX_FILL
from inference.retinaface.postprocess import dets_desde_salidas_modelo
from inference.types import FaceDetections
from utils.image_utils import letterbox_bgr

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # type: ignore[assignment,misc]


class RetinaFaceDetectorRk3568:
    """
    Detector facial RetinaFace (RKNN) para RK3568.

    Preproceso: letterbox BGR -> RGB uint8 (mean/std en el .rknn al export).
    Postproceso: ``dets_desde_salidas_modelo`` (compartido con ONNX).
    """

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        score_pre_nms: float,
    ) -> None:
        if RKNNLite is None:
            raise RuntimeError(
                "rknnlite no instalado. Instala RKNN-Toolkit-Lite2 en la placa "
                "(aarch64), p. ej. rknn_toolkit_lite2-2.3.2-...whl"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe modelo RetinaFace RKNN: {path}"
            )

        self._score_deteccion = float(score_deteccion)
        self._score_pre_nms = float(score_pre_nms)
        self._rknn: RKNNLite | None = RKNNLite()
        if self._rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"load_rknn failed: {path}")
        if self._rknn.init_runtime() != 0:
            self._rknn.release()
            raise RuntimeError(f"init_runtime failed: {path}")

        logging.info("RetinaFace RK3568 (RKNN) cargado: %s", path)

    @classmethod
    def from_settings(cls) -> RetinaFaceDetectorRk3568:
        from configs import settings as s

        return cls(
            model_path=s.retinaface_model_rk3568_path(),
            score_deteccion=s.RETINAFACE_SCORE_DETECCION,
            score_pre_nms=s.RETINAFACE_SCORE_PRE_NMS,
        )

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        if self._rknn is None:
            raise RuntimeError("RetinaFace RK3568: runtime ya liberado")

        h, w = frame_bgr.shape[:2]
        letterbox_img, lb_meta = letterbox_bgr(
            frame_bgr,
            (INPUT_WIDTH, INPUT_HEIGHT),
            LETTERBOX_FILL,
        )
        infer_rgb = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
        if infer_rgb.dtype != np.uint8:
            infer_rgb = infer_rgb.astype(np.uint8)
        input_tensor = np.expand_dims(infer_rgb, axis=0)

        outputs = self._rknn.inference(inputs=[input_tensor])
        if not outputs:
            return FaceDetections.empty()

        dets = dets_desde_salidas_modelo(
            list(outputs),
            img_width=w,
            img_height=h,
            aspect_ratio=lb_meta.aspect_ratio,
            offset_x=lb_meta.offset_x,
            offset_y=lb_meta.offset_y,
            score_deteccion=self._score_deteccion,
            score_pre_nms=self._score_pre_nms,
        )
        if dets.size == 0:
            return FaceDetections.empty()
        return FaceDetections(dets=dets)
