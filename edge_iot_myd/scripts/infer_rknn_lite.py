#!/usr/bin/env python3
"""
Inferencia en la placa MYD-LR3568 con RKNN Lite (NPU). Ejecutar en aarch64 con rknn-toolkit-lite2.

Ejemplo:
  python3 infer_rknn_lite.py --rknn /opt/edge-models/demo_mobilenet_v2/0.1.0/model.rknn
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Inferencia RKNN Lite (demo)")
    parser.add_argument("--rknn", required=True, help="Ruta al fichero .rknn")
    parser.add_argument("--loops", type=int, default=1, help="Repeticiones para medir")
    args = parser.parse_args()

    try:
        import numpy as np
        from rknnlite.api import RKNNLite
    except ImportError:
        print("Instala dependencias en la placa: pip install -r requirements-device.txt", file=sys.stderr)
        return 1

    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(args.rknn)
    if ret != 0:
        print("load_rknn fallo", file=sys.stderr)
        return 1

    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("init_runtime fallo", file=sys.stderr)
        return 1

    # Entrada NHWC uint8 alineada con export demo (224x224x3)
    x = np.zeros((1, 224, 224, 3), dtype=np.uint8)

    for _ in range(args.loops):
        out = rknn_lite.inference(inputs=[x])

    print("inferencia OK")
    if out is not None:
        o = out[0] if isinstance(out, (list, tuple)) else out
        print(f"salida shape: {getattr(o, 'shape', type(o))}")

    rknn_lite.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
