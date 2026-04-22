#!/usr/bin/env python3
"""
Export ONNX -> RKNN para RK3568. Ejecutar en Linux x86_64 con RKNN-Toolkit2 (Docker recomendado).

Ejemplo:
  ./scripts/fetch_demo_onnx.sh
  python3 scripts/export_demo_to_rknn.py --onnx models/demo_mobilenet_v2/mobilenetv2-7.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ONNX a RKNN (target rk3568)")
    parser.add_argument(
        "--onnx",
        required=True,
        help="Ruta al fichero .onnx (p. ej. mobilenetv2-7.onnx)",
    )
    parser.add_argument(
        "--out",
        default="models/demo_mobilenet_v2/model.rknn",
        help="Ruta de salida del .rknn",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Activar cuantizacion INT8 (requiere dataset de calibracion; no implementado en demo)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"No existe ONNX: {args.onnx}", file=sys.stderr)
        return 1

    try:
        from rknn.api import RKNN
    except ImportError:
        print("Instala RKNN-Toolkit2 en este entorno (ver docker/README.md).", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    rknn = RKNN(verbose=True)

    # Preproceso tipico ImageNet / MobileNet; ajustar si cambias el ONNX
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[128.0, 128.0, 128.0]],
        target_platform="rk3568",
    )

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("load_onnx fallo", file=sys.stderr)
        return 1

    if args.quantize:
        print("Cuantizacion: usar flujo con dataset en produccion; demo sin calibracion.", file=sys.stderr)
        return 1

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("build fallo", file=sys.stderr)
        return 1

    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print("export_rknn fallo", file=sys.stderr)
        return 1

    rknn.release()

    try:
        from importlib.metadata import version

        toolkit_ver = version("rknn_toolkit2")
    except Exception:
        toolkit_ver = "desconocida"

    digest = _sha256_file(args.out)
    print(f"OK: {args.out}")
    print(f"sha256: {digest}")
    print(f"rknn_toolkit_version: {toolkit_ver}")
    print("Actualiza models/demo_mobilenet_v2/manifest.json con sha256 y rknn_toolkit_version.")

    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "manifest.json"
    )
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        man["artifact"]["sha256"] = digest
        man["rknn_toolkit_version"] = str(toolkit_ver)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
            f.write("\n")
        print(f"Manifest actualizado: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
