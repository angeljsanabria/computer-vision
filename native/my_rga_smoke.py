#!/usr/bin/env python3
"""Smoke test post pip install de my_rga (ejecutar en placa RK3568)."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import my_rga
    except ImportError as exc:
        print("FAIL: import my_rga:", exc)
        return 1

    print("OK import:", my_rga.__doc__ or "(sin docstring)")
    print("API:", [x for x in dir(my_rga) if not x.startswith("_")])

    import numpy as np

    z = np.zeros((48, 64, 3), dtype=np.uint8)
    out, used = my_rga.resize_bgr(z, 32, 32)
    print("resize_bgr:", out.shape, "used_rga=", used)

    canvas, scale, px, py, used_lb = my_rga.letterbox_bgr(z, 32, 32, 114)
    print("letterbox_bgr:", canvas.shape, scale, px, py, "used_rga=", used_lb)

    rgb, used_rgb = my_rga.bgr_to_rgb(z)
    print("bgr_to_rgb:", rgb.shape, "used_rga=", used_rgb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
