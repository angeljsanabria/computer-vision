"""
Export nozzle RKNN v3 (RK3568) con hybrid INT8.

Problema v2 INT8: cuantizacion plena de YOLOv8 Ultralytics deja output0 en cero
(best_score=0.000 en placa). Solucion: hybrid quant (backbone INT8 + output0 FP16).

Patron alineado a RetinaFace/FaceMesh (mean 0 / std 255, calib representativa)
+ hybrid_quant del SDK (examples/functions/hybrid_quant/).

Flujo:
  python yolo_train/prepare_nozzle_calib_v3.py
  python yolo_train/exp_yolov8n_nozzle_rknn_v3.py

Salida:
  Yolo-Weights/yolov8n_nozzle_v3.rknn
  models/yolov8n_nozzle_v3.rknn
  Yolo-Weights/yolov8n_nozzle.rknn (alias)

WSL + rknn-toolkit2 2.3.2 (venv cp311 x86_64).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nozzle_config as nc  # noqa: E402
from prepare_nozzle_calib_v3 import build_calib_v3  # noqa: E402

from rknn.api import RKNN

TARGET_PLATFORM = "rk3568"


def _validate_dataset(dataset_path: Path) -> None:
    missing = [
        line.strip()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not (SCRIPT_DIR / line.strip()).is_file()
    ]
    if missing:
        raise SystemExit(
            "Imagenes de calibracion no encontradas (paths relativos a yolo_train/):\n"
            + "\n".join(missing[:10])
            + (f"\n... y {len(missing) - 10} mas" if len(missing) > 10 else "")
        )


def _dataset_with_abs_paths(dataset_path: Path, out_path: Path) -> Path:
    """RKNN resuelve paths del dataset respecto al cwd; usamos rutas absolutas."""
    lines: list[str] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        abs_img = (SCRIPT_DIR / rel).resolve()
        if not abs_img.is_file():
            raise SystemExit(f"Imagen calibracion no encontrada: {abs_img}")
        lines.append(str(abs_img))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _ensure_calib() -> Path:
    path = nc.RKNN_CALIB_DATASET.resolve()
    if path.is_file() and path.stat().st_size > 0:
        _validate_dataset(path)
        return path
    print("--> Generando calibracion v3 (640 stretch)...")
    n = build_calib_v3(
        dataset_root=nc.DATASET_ROOT.resolve(),
        out_dir=nc.RKNN_CALIB_DIR.resolve(),
        dataset_txt=path,
        max_images=nc.RKNN_CALIB_MAX_IMAGES,
        seed=42,
    )
    print(f"    {n} imagenes -> {path}")
    return path


def _patch_quant_cfg(cfg_path: Path, fp16_nodes: tuple[str, ...]) -> None:
    """Fuerza custom_quantize_layers con los nodos FP16 (fix ultralytics#23340)."""
    import re

    text = cfg_path.read_text(encoding="utf-8")
    entries = "\n".join(f"    {node}: float16" for node in fp16_nodes)
    replacement = f"custom_quantize_layers:\n{entries}"

    # Reemplaza bloque completo (vacio {}, o con entradas previas) hasta quantize_parameters
    pattern = re.compile(
        r"^custom_quantize_layers:.*?(?=^quantize_parameters:)",
        flags=re.M | re.S,
    )
    if pattern.search(text):
        text = pattern.sub(replacement + "\n", text, count=1)
    else:
        text = replacement + "\n\n" + text

    cfg_path.write_text(text, encoding="utf-8")
    print("    Parche FP16 custom_quantize_layers:", ", ".join(fp16_nodes))


def _rknn_config(rknn: RKNN, *, use_mmse: bool = False) -> None:
    kwargs: dict = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": TARGET_PLATFORM,
        "quantized_method": "channel",
    }
    if use_mmse:
        kwargs["quantized_algorithm"] = "mmse"
    rknn.config(**kwargs)


def _run_hybrid_step2(
    *,
    cfg_path: Path,
    model_path: Path,
    data_path: Path,
    out_rknn: Path,
    fp16_nodes: tuple[str, ...],
) -> None:
    if not model_path.is_file() or not data_path.is_file():
        raise SystemExit(
            f"Faltan artefactos hybrid en {cfg_path.parent}\n"
            "Corre export completo (sin --step2-only)."
        )
    print("--> patch quantization.cfg (output FP16)")
    _patch_quant_cfg(cfg_path, fp16_nodes)

    old_cwd = Path.cwd()
    os.chdir(cfg_path.parent)
    try:
        rknn = RKNN(verbose=True)
        print("--> hybrid_quantization_step2")
        ret = rknn.hybrid_quantization_step2(
            model_input=str(model_path),
            data_input=str(data_path),
            model_quantization_cfg=str(cfg_path),
        )
        if ret != 0:
            raise SystemExit(f"hybrid_quantization_step2 failed: {ret}")

        print("--> export_rknn")
        ret = rknn.export_rknn(str(out_rknn))
        if ret != 0:
            raise SystemExit(f"export_rknn failed: {ret}")
        rknn.release()
    finally:
        os.chdir(old_cwd)


def export_hybrid_v3(
    dataset_path: Path, *, use_mmse: bool = False, step2_only: bool = False
) -> Path:
    onnx_path = nc.ONNX_RKNN_SOURCE.resolve()
    if not onnx_path.is_file():
        raise SystemExit(
            f"No se encuentra ONNX fuente: {onnx_path}\n"
            "Usa el v2 entrenado o: python yolo_train/export_nozzle_onnx.py"
        )

    build_dir = nc.RKNN_BUILD_DIR.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    stem = onnx_path.stem

    cfg_path = build_dir / f"{stem}.quantization.cfg"
    model_path = build_dir / f"{stem}.model"
    data_path = build_dir / f"{stem}.data"
    dataset_abs = build_dir / "calib_dataset_abs.txt"
    out_rknn = nc.RKNN_VERSIONED.resolve()

    if step2_only:
        _run_hybrid_step2(
            cfg_path=cfg_path,
            model_path=model_path,
            data_path=data_path,
            out_rknn=out_rknn,
            fp16_nodes=nc.RKNN_HYBRID_FP16_NODES,
        )
        return out_rknn

    _dataset_with_abs_paths(dataset_path, dataset_abs)

    old_cwd = Path.cwd()
    os.chdir(build_dir)
    try:
        rknn = RKNN(verbose=True)
        print(f"--> config (channel, target rk3568, mmse={use_mmse})")
        _rknn_config(rknn, use_mmse=use_mmse)

        print("--> load_onnx")
        print("    ", onnx_path)
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise SystemExit(f"load_onnx failed: {ret}")

        print("--> hybrid_quantization_step1 (proposal=False)")
        ret = rknn.hybrid_quantization_step1(
            dataset=str(dataset_abs),
            proposal=False,
        )
        if ret != 0:
            raise SystemExit(f"hybrid_quantization_step1 failed: {ret}")

        if not cfg_path.is_file():
            raise SystemExit(f"No se genero {cfg_path}")

        rknn.release()

        _run_hybrid_step2(
            cfg_path=cfg_path,
            model_path=model_path,
            data_path=data_path,
            out_rknn=out_rknn,
            fp16_nodes=nc.RKNN_HYBRID_FP16_NODES,
        )
    finally:
        os.chdir(old_cwd)

    return out_rknn


def export_full_int8(dataset_path: Path, *, use_mmse: bool = False) -> Path:
    """Fallback INT8 pleno (debug; puede dar cls=0 en YOLOv8 Ultralytics)."""
    onnx_path = nc.ONNX_RKNN_SOURCE.resolve()
    out_rknn = nc.RKNN_VERSIONED.resolve()
    build_dir = nc.RKNN_BUILD_DIR.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    dataset_abs = build_dir / "calib_dataset_abs.txt"
    _dataset_with_abs_paths(dataset_path, dataset_abs)

    rknn = RKNN(verbose=True)
    _rknn_config(rknn, use_mmse=use_mmse)
    print("--> load_onnx", onnx_path)
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")
    print("--> build full INT8")
    ret = rknn.build(do_quantization=True, dataset=str(dataset_abs))
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")
    ret = rknn.export_rknn(str(out_rknn))
    if ret != 0:
        raise SystemExit(f"export failed: {ret}")
    rknn.release()
    return out_rknn


def _deploy(out_rknn: Path) -> None:
    nc.RKNN_DEPLOY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(out_rknn), str(nc.RKNN_LATEST))
    shutil.copy2(str(out_rknn), str(nc.RKNN_DEPLOY))
    size_mb = out_rknn.stat().st_size / (1024 * 1024)
    print(f"OK -> {out_rknn} ({size_mb:.2f} MB)")
    print(f"OK -> {nc.RKNN_LATEST} (alias)")
    print(f"OK -> {nc.RKNN_DEPLOY} (runtime placa)")
    print(
        "En placa:\n"
        f"  NOZZLE_MODEL_RK3568={nc.RKNN_DEPLOY.as_posix()}\n"
        "  NOZZLE_SCORE_DETECCION=0.30"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nozzle RKNN v3 hybrid INT8.")
    parser.add_argument(
        "--full-int8",
        action="store_true",
        help="INT8 pleno (no recomendado; puede zerar output0).",
    )
    parser.add_argument(
        "--mmse",
        action="store_true",
        help="Usar mmse (mas lento y RAM; default: normal).",
    )
    parser.add_argument(
        "--step2-only",
        action="store_true",
        help="Reutiliza artefactos step1 en rknn_build_v3/ (patch FP16 + step2).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset.txt calibracion (default rknn_nozzle_v3_dataset.txt).",
    )
    args = parser.parse_args()

    dataset_path = (
        args.dataset.resolve()
        if args.dataset is not None
        else _ensure_calib()
    )
    if not dataset_path.is_file():
        raise SystemExit(
            f"No existe {dataset_path}\n"
            "Corre: python yolo_train/prepare_nozzle_calib_v3.py"
        )
    _validate_dataset(dataset_path)

    print(f"RKNN export:  {nc.RKNN_EXPORT_VERSION}")
    print(f"ONNX fuente:  {nc.ONNX_RKNN_SOURCE}")
    print(f"Calibracion:  {dataset_path}")
    print(f"Modo:         {'full INT8' if args.full_int8 else 'hybrid INT8 + output0 FP16'}")
    print(f"Algorithm:    {'mmse' if args.mmse else 'normal (default)'}")

    if args.full_int8:
        out = export_full_int8(dataset_path, use_mmse=args.mmse)
    else:
        out = export_hybrid_v3(
            dataset_path,
            use_mmse=args.mmse,
            step2_only=args.step2_only,
        )

    _deploy(out)


if __name__ == "__main__":
    main()
