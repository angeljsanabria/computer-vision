"""Detector Bidon/Pico YOLOv8 RKNN en RK3568 (RKNN Lite)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.nozzle_bidon.constants import INPUT_SIZE
from inference.nozzle_bidon.postprocess import (
    postprocess_yolov8_ultralytics,
    scale_boxes_stretch,
)
from inference.nozzle_bidon.preprocess import stretch_bgr_to_rknn_input
from inference.nozzle_bidon.types import NozzleBidonDetections

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # type: ignore[assignment,misc]


class NozzleBidonDetectorRk3568:
    """
    YOLOv8n Bidon/Pico (2 clases) en NPU Rockchip.

    Preproceso: resize stretch INPUT_SIZE, RGB uint8 NHWC (mean 0 / std 255).
    """

    def __init__(
        self,
        model_path: str | Path,
        score_deteccion: float,
        nms_iou: float,
    ) -> None:
        if RKNNLite is None:
            raise RuntimeError(
                "rknnlite no instalado. Instala RKNN-Toolkit-Lite2 en la placa "
                "(aarch64), p. ej. rknn_toolkit_lite2-2.3.2-...whl"
            )
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe modelo nozzle_bidon RKNN: {path}")

        self._score = float(score_deteccion)
        self._nms_iou = float(nms_iou)
        self._rknn: RKNNLite | None = RKNNLite()
        if self._rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"load_rknn failed: {path}")
        if self._rknn.init_runtime() != 0:
            self._rknn.release()
            raise RuntimeError(f"init_runtime failed: {path}")

        logging.debug("NozzleBidon RK3568 (RKNN) cargado: %s", path)

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray) -> NozzleBidonDetections:
        if self._rknn is None:
            raise RuntimeError("NozzleBidon RK3568: runtime ya liberado")

        h, w = frame_bgr.shape[:2]
        inp = stretch_bgr_to_rknn_input(frame_bgr)
        outputs = self._rknn.inference(inputs=[inp])
        if not outputs:
            return NozzleBidonDetections.empty()

        pred = np.asarray(outputs[0])
        xyxy, scores, class_ids = postprocess_yolov8_ultralytics(
            pred, self._score, self._nms_iou
        )
        if xyxy is None or scores is None or class_ids is None:
            return NozzleBidonDetections.empty()

        xyxy = scale_boxes_stretch(xyxy, w, h, INPUT_SIZE)
        dets = np.column_stack(
            [xyxy, scores, class_ids.astype(np.float32)]
        ).astype(np.float32)
        return NozzleBidonDetections(dets=dets)
