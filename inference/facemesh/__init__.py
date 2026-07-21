"""FaceMesh 468: landmarks densos PC (ONNX) / RK3568 (RKNN) y utilidades."""
from inference.facemesh.constants import (
    INPUT_HW,
    INPUT_SIZE,
    LANDMARK_DIM,
    NUM_LANDMARKS,
)
from inference.facemesh.estimator_pc import FaceMeshEstimatorPc
from inference.facemesh.estimator_rk3568 import FaceMeshEstimatorRk3568
from inference.facemesh.from_detection import estimate_from_det
from inference.facemesh.postprocess import landmarks_mesh_to_frame
from inference.facemesh.preprocess import (
    bgr192_to_onnx_nchw,
    bgr192_to_rknn_nhwc,
    crop_to_bgr192,
)

__all__ = [
    "INPUT_HW",
    "INPUT_SIZE",
    "LANDMARK_DIM",
    "NUM_LANDMARKS",
    "FaceMeshEstimatorPc",
    "FaceMeshEstimatorRk3568",
    "bgr192_to_onnx_nchw",
    "bgr192_to_rknn_nhwc",
    "crop_to_bgr192",
    "estimate_from_det",
    "landmarks_mesh_to_frame",
]
