"""RetinaFace: deteccion facial (PC ONNX / RK3568 RKNN)."""
from inference.retinaface.constants import (
    INPUT_HEIGHT,
    INPUT_HW,
    INPUT_WIDTH,
    LETTERBOX_FILL,
    MEAN_BGR,
)
from inference.retinaface.detector_pc import RetinaFaceDetectorPc
from inference.retinaface.detector_rk3568 import RetinaFaceDetectorRk3568
from inference.retinaface.postprocess import dets_desde_salidas_modelo
from inference.retinaface.select_best import mejores_caras, rank_cara, seleccionar_caras

__all__ = [
    "INPUT_HEIGHT",
    "INPUT_HW",
    "INPUT_WIDTH",
    "LETTERBOX_FILL",
    "MEAN_BGR",
    "RetinaFaceDetectorPc",
    "RetinaFaceDetectorRk3568",
    "dets_desde_salidas_modelo",
    "rank_cara",
    "mejores_caras",
    "seleccionar_caras",
]
