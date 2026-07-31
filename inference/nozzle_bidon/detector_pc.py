"""Detector Bidon/Pico YOLOv8 ONNX en PC via ONNX Runtime."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.nozzle_bidon.constants import LETTERBOX_FILL
from inference.nozzle_bidon.postprocess import postprocess_yolov8_ultralytics
from inference.nozzle_bidon.preprocess import (
    letterbox_bgr_for_onnx,
    letterbox_to_nchw_float01,
    scale_boxes_letterbox,
)
from inference.nozzle_bidon.types import NozzleBidonDetections

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment,misc]


class NozzleBidonDetectorPc:
    """
    YOLOv8 ONNX Bidon/Pico en PC.

    Preproceso letterbox input_size; salida tipica (1, 4+nc, N) con nc=2.
    """

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        nms_iou: float,
        input_size: int,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime no instalado. pip install onnxruntime"
            )
        if input_size < 32:
            raise ValueError(f"input_size invalido: {input_size}")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo nozzle_bidon ONNX: {path}")

        self._input_size = int(input_size)
        self._score = float(score_deteccion)
        self._nms_iou = float(nms_iou)
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        logging.info(
            "NozzleBidon PC (ONNX Runtime) cargado: %s input=%d",
            path,
            self._input_size,
        )
        self._warmup()

    def _warmup(self) -> None:
        sz = self._input_size
        dummy = np.zeros((1, 3, sz, sz), dtype=np.float32)
        try:
            self._session.run(None, {self._input_name: dummy})
        except Exception as exc:
            logging.warning(
                "[NozzleBidon] warmup ONNX fallo (se reintenta en detect): %s", exc
            )

    def detect(self, frame_bgr: np.ndarray) -> NozzleBidonDetections:
        h, w = frame_bgr.shape[:2]
        canvas, meta = letterbox_bgr_for_onnx(
            frame_bgr, LETTERBOX_FILL, self._input_size
        )
        feed = letterbox_to_nchw_float01(canvas)
        outputs = self._session.run(None, {self._input_name: feed})
        if not outputs:
            return NozzleBidonDetections.empty()

        pred = np.asarray(outputs[0])
        xyxy, scores, class_ids = postprocess_yolov8_ultralytics(
            pred, self._score, self._nms_iou
        )
        if xyxy is None or scores is None or class_ids is None:
            return NozzleBidonDetections.empty()

        xyxy = scale_boxes_letterbox(xyxy, meta, w, h)
        dets = np.column_stack(
            [xyxy, scores, class_ids.astype(np.float32)]
        ).astype(np.float32)
        return NozzleBidonDetections(dets=dets)

    def release(self) -> None:
        pass
