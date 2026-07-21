"""Inspeccion rapida del ONNX FaceMesh (I/O y tamano)."""
from __future__ import annotations

from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT.parent / "models_onnx" / "face_mesh_192x192.onnx"
NUM_LANDMARKS = 468
LANDMARK_DIM = 3


def _shape(dims):
    return [d.dim_value or d.dim_param for d in dims]


def main() -> None:
    if not ONNX_PATH.is_file():
        raise SystemExit(f"No existe: {ONNX_PATH}")

    model = onnx.load(str(ONNX_PATH))
    print("ONNX:", ONNX_PATH)
    print("Size MB:", round(ONNX_PATH.stat().st_size / (1024 * 1024), 2))

    for inp in model.graph.input:
        t = inp.type.tensor_type
        print("input:", inp.name, _shape(t.shape.dim))

    for out in model.graph.output:
        t = out.type.tensor_type
        shape = _shape(t.shape.dim)
        print("output:", out.name, shape)
        flat = 1
        for s in shape:
            if isinstance(s, int):
                flat *= s
        if flat == NUM_LANDMARKS * LANDMARK_DIM:
            print(f"  -> {NUM_LANDMARKS} landmarks x {LANDMARK_DIM}")
        elif out.name == "score":
            print("  -> score auxiliar (ignorado en runtime)")


if __name__ == "__main__":
    main()
