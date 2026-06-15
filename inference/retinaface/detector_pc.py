"""RetinaFace en PC via ONNX Runtime."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.retinaface.constants import INPUT_HEIGHT, INPUT_WIDTH, LETTERBOX_FILL, MEAN_BGR
from inference.retinaface.postprocess import dets_desde_salidas_modelo
from inference.types import FaceDetections
from utils.image_utils import letterbox_bgr

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment,misc]


def _preprocess_canvas_for_onnx(inp0_meta, canvas_bgr: np.ndarray) -> np.ndarray:
    """Tensor de entrada segun firma ONNX (NCHW o NHWC). Port de RetinaFace_from_cam.py."""
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape

    nchw = False
    nhwc = False
    if len(dims) == 4 and dims[1] == 3:
        nchw = True
    elif len(dims) == 4 and dims[-1] == 3:
        nhwc = True
    elif len(dims) == 4 and dims[1] == INPUT_WIDTH:
        nhwc = True
    else:
        nchw = True

    if nhwc:
        x32 -= MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)

    if nchw:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= MEAN_BGR.reshape(1, 3, 1, 1)
        return feed

    raise RuntimeError("Forma de entrada ONNX no soportada: " + repr(dims))


class RetinaFaceDetectorPc:
    """Detector facial RetinaFace (ONNX) para desarrollo en PC."""

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        score_pre_nms: float,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime no instalado. pip install onnxruntime"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo RetinaFace ONNX: {path}")

        self._score_deteccion = float(score_deteccion)
        self._score_pre_nms = float(score_pre_nms)
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        self._inp0_meta = inputs[0]
        logging.info("RetinaFace PC (ONNX) cargado: %s", path)

    @classmethod
    def from_settings(cls) -> RetinaFaceDetectorPc:
        from configs import settings as s

        return cls(
            model_path=s.retinaface_model_pc_path(),
            score_deteccion=s.RETINAFACE_SCORE_DETECCION,
            score_pre_nms=s.RETINAFACE_SCORE_PRE_NMS,
        )

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        h, w = frame_bgr.shape[:2]
        letterbox_img, lb_meta = letterbox_bgr(
            frame_bgr,
            (INPUT_WIDTH, INPUT_HEIGHT),
            LETTERBOX_FILL,
        )
        tensor = _preprocess_canvas_for_onnx(self._inp0_meta, letterbox_img)
        ort_outputs = self._session.run(None, {self._input_name: tensor})
        dets = dets_desde_salidas_modelo(
            list(ort_outputs),
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
