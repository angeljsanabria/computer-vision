"""
Exporta Retinaface-Models/RetinaFace_mobile320.onnx -> Retinaface-Models/RetinaFace_mobile320.rknn.

Basado en Retinaface-Models/convert.py, pero con rutas absolutas al repo
para que funcione desde cualquier directorio.
"""
from __future__ import annotations

from pathlib import Path

from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "Retinaface-Models" / "RetinaFace_mobile320.onnx"
RKNN_PATH = ROOT / "Retinaface-Models" / "RetinaFace_mobile320.rknn"
DATASET_PATH = ROOT / "rknn_model_zoo" / "examples" / "RetinaFace" / "model" / "dataset.txt"

# RetinaFace del zoo usa normalizacion tipo Caffe/BGR:
# mean=[104,117,123], std=[1,1,1]
MEAN_VALUES = [[104, 117, 123]]
STD_VALUES = [[1, 1, 1]]
TARGET_PLATFORM = "rk3568"
DO_QUANTIZATION = True


def main() -> None:
    if not ONNX_PATH.is_file():
        raise SystemExit(f"No existe ONNX: {ONNX_PATH}")
    if DO_QUANTIZATION and not DATASET_PATH.is_file():
        raise SystemExit(
            "No existe dataset de calibracion para INT8: "
            f"{DATASET_PATH}"
        )

    rknn = RKNN(verbose=True)

    print("--> config")
    rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=TARGET_PLATFORM,
    )

    print("--> load_onnx")
    print("    ", ONNX_PATH)
    ret = rknn.load_onnx(model=str(ONNX_PATH))
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")

    print("--> build")
    ret = rknn.build(
        do_quantization=DO_QUANTIZATION,
        dataset=str(DATASET_PATH) if DO_QUANTIZATION else None,
    )
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")

    print("--> export_rknn")
    ret = rknn.export_rknn(str(RKNN_PATH))
    if ret != 0:
        raise SystemExit(f"export failed: {ret}")

    rknn.release()
    print("OK ->", RKNN_PATH)


if __name__ == "__main__":
    main()
