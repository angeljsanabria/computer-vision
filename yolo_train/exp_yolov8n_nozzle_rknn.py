"""
Exporta Yolo-Weights/yolov8n_nozzle_<version>.onnx -> yolov8n_nozzle_<version>.rknn (RK3568).

Config: yolo_train/nozzle_config.py

Calibracion INT8 (default):
  python yolo_train/gen_nozzle_rknn_dataset.py
  python yolo_train/exp_yolov8n_nozzle_rknn.py

Sin cuantizar (debug / comparar latencia FP):
  python yolo_train/exp_yolov8n_nozzle_rknn.py --no-quant

Uso en PC/WSL con rknn-toolkit2 2.3.2 (venv x86_64).

Salida:
  Yolo-Weights/yolov8n_<version>.rknn  (versionado; ver nozzle_config)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402
from gen_nozzle_rknn_dataset import build_dataset_txt  # noqa: E402

from rknn.api import RKNN

ONNX_PATH = nc.ONNX_VERSIONED
RKNN_PATH = nc.RKNN_VERSIONED
RKNN_CALIB_DATASET = nc.RKNN_CALIB_DATASET
TARGET_PLATFORM = "rk3568"
DO_QUANTIZATION_DEFAULT = True


def _ensure_calib_dataset() -> Path:
    path = RKNN_CALIB_DATASET.resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    print("--> Generando dataset de calibracion INT8...")
    n = build_dataset_txt(
        dataset_root=nc.DATASET_ROOT.resolve(),
        out_path=path,
        max_images=nc.RKNN_CALIB_MAX_IMAGES,
        seed=42,
    )
    print(f"    {n} imagenes -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nozzle ONNX -> RKNN (RK3568).")
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Desactiva INT8 (build FP, mas lento en placa).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset.txt calibracion (default: yolo_train/rknn_nozzle_dataset.txt).",
    )
    args = parser.parse_args()

    do_quant = DO_QUANTIZATION_DEFAULT and not args.no_quant

    if not ONNX_PATH.is_file():
        raise SystemExit(
            f"No se encuentra {ONNX_PATH}\n"
            "Exporta antes: python yolo_train/export_nozzle_onnx.py"
        )

    dataset_path: Path | None = None
    if do_quant:
        dataset_path = (
            args.dataset.resolve()
            if args.dataset is not None
            else _ensure_calib_dataset()
        )
        if not dataset_path.is_file():
            raise SystemExit(
                f"No existe dataset de calibracion: {dataset_path}\n"
                "Genera uno: python yolo_train/gen_nozzle_rknn_dataset.py"
            )

    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET_PLATFORM,
    )

    print(f"Version:      {nc.NOZZLE_VERSION}")
    print(f"Quant INT8:   {do_quant}")
    if do_quant and dataset_path is not None:
        print(f"Calibracion:  {dataset_path}")
        print("Nota: paths en dataset.txt son relativos a yolo_train/.")
    print("--> load_onnx")
    print("    ", ONNX_PATH)
    ret = rknn.load_onnx(model=str(ONNX_PATH))
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")

    print("--> build")
    ret = rknn.build(
        do_quantization=do_quant,
        dataset=str(dataset_path) if do_quant and dataset_path else None,
    )
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")

    print("--> export_rknn")
    ret = rknn.export_rknn(str(RKNN_PATH))
    if ret != 0:
        raise SystemExit(f"export failed: {ret}")

    rknn.release()
    print("OK ->", RKNN_PATH)
    if do_quant:
        print(
            f"Despliega a placa: copiar a {nc.RKNN_DEPLOY} "
            "(o NOZZLE_MODEL_RK3568 en settings)."
        )


if __name__ == "__main__":
    main()
