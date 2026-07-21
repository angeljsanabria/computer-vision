"""
Exporta models_onnx/face_mesh_192x192.onnx -> facemesh-models/face_mesh_192x192.rknn.

Entrada RGB 192x192; en placa uint8 NHWC + mean/std en rknn.config (equivale a /255).
Patron: mobilenet_modelos/exp_mobilefacenet_rknn.py y export_models/exp_retinaface_rknn.py.

Requisitos (WSL/Linux x86_64 + RKNN-Toolkit2 cp311):
  - ../models_onnx/face_mesh_192x192.onnx
  - dataset.txt + calib/*.jpg (si DO_QUANTIZATION=True)

Ejemplo:
  cd facemesh-models
  python prepare_calib.py
  python exp_facemesh_rknn.py
"""
from __future__ import annotations

from pathlib import Path

from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT.parent / "models_onnx" / "face_mesh_192x192.onnx"
RKNN_PATH = ROOT / "face_mesh_192x192.rknn"
DATASET_PATH = ROOT / "dataset.txt"

# RGB uint8 -> float [0,1]: (x - 0) / 255  (igual que inference/facemesh/preprocess.py ONNX)
MEAN_VALUES = [[0, 0, 0]]
STD_VALUES = [[255, 255, 255]]

TARGET_PLATFORM = "rk3568"
DO_QUANTIZATION = True


def main() -> None:
    if not ONNX_PATH.is_file():
        raise SystemExit(f"No existe ONNX: {ONNX_PATH}")
    if DO_QUANTIZATION and not DATASET_PATH.is_file():
        raise SystemExit(
            "No existe dataset de calibracion INT8: "
            f"{DATASET_PATH} (correr prepare_calib.py)"
        )
    if DO_QUANTIZATION:
        missing = [
            line.strip()
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not (ROOT / line.strip()).is_file()
        ]
        if missing:
            raise SystemExit(
                "Imagenes del dataset no encontradas en facemesh-models/: "
                + ", ".join(missing)
            )

    rknn = RKNN(verbose=True)

    print("--> config (RGB /255, target %s)" % TARGET_PLATFORM)
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

    print("--> build (do_quantization=%s)" % DO_QUANTIZATION)
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
    print("Copiar a runtime: cp", RKNN_PATH.name, "../models/")


if __name__ == "__main__":
    main()
