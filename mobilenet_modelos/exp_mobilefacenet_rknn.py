"""
Exporta MobileFaceNet.onnx (+ MobileFaceNet.onnx.data) -> MobileFaceNet.rknn.

Entrada RGB 112x112; mean/std ImageNet en rknn.config (uint8 RGB en placa).
Equivalente a export_models/face_embedding_from_image.py (preproceso embed).

Requisitos (todo en esta carpeta mobilenet_modelos/):
  - MobileFaceNet.onnx, MobileFaceNet.onnx.data
  - dataset.txt + calib/*.jpg (calibracion INT8)
  - RKNN-Toolkit2 en PC Linux, target rk3568

Ejemplo:
  cd mobilenet_modelos
  python exp_mobilefacenet_rknn.py

Sugerido: primero DO_QUANTIZATION=False; luego True con mas JPG en calib/.
"""
from __future__ import annotations

from pathlib import Path

from rknn.api import RKNN

DIR = Path(__file__).resolve().parent
ONNX_PATH = DIR / "MobileFaceNet.onnx"
RKNN_PATH = DIR / "MobileFaceNet.rknn"
DATASET_PATH = DIR / "dataset.txt"

# ImageNet (foamliu / torchvision): (x/255 - mean) / std sobre RGB uint8
# mean * 255, std * 255 para rknn.config
MEAN_VALUES = [[123.675, 116.28, 103.53]]
STD_VALUES = [[58.395, 57.12, 57.375]]

TARGET_PLATFORM = "rk3568"
DO_QUANTIZATION = True


def main() -> None:
    if not ONNX_PATH.is_file():
        raise SystemExit(f"No existe ONNX: {ONNX_PATH}")
    data_file = DIR / "MobileFaceNet.onnx.data"
    if not data_file.is_file():
        raise SystemExit(
            f"No existe peso externo ONNX: {data_file} "
            "(debe estar junto a MobileFaceNet.onnx)"
        )
    if DO_QUANTIZATION and not DATASET_PATH.is_file():
        raise SystemExit(
            "No existe dataset de calibracion INT8: "
            f"{DATASET_PATH}"
        )
    if DO_QUANTIZATION:
        missing = [
            line.strip()
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not (DIR / line.strip()).is_file()
        ]
        if missing:
            raise SystemExit(
                "Imagenes del dataset no encontradas en mobilenet_modelos/: "
                + ", ".join(missing)
            )

    rknn = RKNN(verbose=True)

    print("--> config (RGB ImageNet, target %s)" % TARGET_PLATFORM)
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


if __name__ == "__main__":
    main()
