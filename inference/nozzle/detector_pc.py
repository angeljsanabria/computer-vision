"""Detector fuel nozzle YOLOv8 ONNX en PC via ONNX Runtime (sin Ultralytics)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.nozzle.constants import INPUT_SIZE, LETTERBOX_FILL
from inference.nozzle.postprocess import postprocess_yolov8_ultralytics
from inference.nozzle.preprocess import (
    letterbox_bgr_for_onnx,
    letterbox_to_nchw_float01,
    scale_boxes_letterbox,
)
from inference.nozzle.types import NozzleDetections

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment,misc]


class NozzleDetectorPc:
    """
    YOLOv8 ONNX fine-tune nozzle en PC.

    Carga eager en __init__ (sin lazy load de Ultralytics en el primer frame).
    Preproceso letterbox 640 + salida tipica (1, 5, 8400).
    """

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        nms_iou: float,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime no instalado. pip install onnxruntime"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo nozzle ONNX: {path}")

        self._score = float(score_deteccion)
        self._nms_iou = float(nms_iou)
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        logging.info("Nozzle PC (ONNX Runtime) cargado: %s", path)
        self._warmup()

    def _warmup(self) -> None:
        """Reserva sesion ORT en arranque; evita sorpresa en el primer frame del loop."""
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        try:
            self._session.run(None, {self._input_name: dummy})
        except Exception as exc:
            logging.warning("[Nozzle] warmup ONNX fallo (se reintenta en detect): %s", exc)

    def detect(self, frame_bgr: np.ndarray) -> NozzleDetections:
        h, w = frame_bgr.shape[:2]
        canvas, meta = letterbox_bgr_for_onnx(frame_bgr, LETTERBOX_FILL)
        feed = letterbox_to_nchw_float01(canvas)
        outputs = self._session.run(None, {self._input_name: feed})
        if not outputs:
            return NozzleDetections.empty()

        pred = np.asarray(outputs[0])
        xyxy, scores = postprocess_yolov8_ultralytics(
            pred, self._score, self._nms_iou
        )
        if xyxy is None or scores is None:
            return NozzleDetections.empty()

        xyxy = scale_boxes_letterbox(xyxy, meta, w, h)
        dets = np.column_stack([xyxy, scores]).astype(np.float32)
        return NozzleDetections(dets=dets)

    def release(self) -> None:
        pass
