#!/usr/bin/env python3
"""
Compara salidas RGA vs OpenCV en placa RK3568.

Uso en placa:
  INFERENCE_BACKEND=rk3568 USE_RGA=true python scripts/validate_rga_vs_opencv.py

Requiere my_rga instalado (pip install native/wheels/my_rga-*.whl en placa).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _max_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def main() -> int:
    backend = os.getenv("INFERENCE_BACKEND", "pc").lower()
    if backend != "rk3568":
        print("Skip: INFERENCE_BACKEND debe ser rk3568 (actual: %s)" % backend)
        return 0

    try:
        import my_rga
    except ImportError:
        print("ERROR: my_rga no instalado. pip install rknn-toolkit-lite/my_rga-*.whl")
        return 1

    from utils.image_backend import opencv_letterbox_bgr, opencv_resize
    from utils.image_utils import bgr_to_rgb, letterbox_bgr, resize_frame

    os.environ["USE_RGA"] = "true"
    rng = np.random.default_rng(0)

    cases = [
        ("640x480", (480, 640, 3)),
        ("1080p", (1080, 1920, 3)),
    ]

    failed = False
    for name, shape in cases:
        frame = rng.integers(0, 256, size=shape, dtype=np.uint8)

        cv_stretch = opencv_resize(frame, (640, 640), cv2.INTER_LINEAR)
        rga_stretch = resize_frame(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        d_stretch = _max_diff(cv_stretch, rga_stretch)
        print("[%s] stretch 640x640 max_diff=%d" % (name, d_stretch))
        if d_stretch > 2:
            failed = True

        cv_lb, _, _, _ = opencv_letterbox_bgr(frame, (320, 320), 114)
        rga_lb, _meta = letterbox_bgr(frame, (320, 320), 114)
        d_lb = _max_diff(cv_lb, rga_lb)
        print("[%s] letterbox 320 max_diff=%d" % (name, d_lb))
        if d_lb > 2:
            failed = True

        cv_rgb = cv2.cvtColor(cv_lb, cv2.COLOR_BGR2RGB)
        rga_rgb = bgr_to_rgb(rga_lb)
        d_rgb = _max_diff(cv_rgb, rga_rgb)
        print("[%s] bgr_to_rgb max_diff=%d" % (name, d_rgb))
        if d_rgb > 0:
            failed = True

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, used = my_rga.resize_bgr(img, 320, 320)
    print("my_rga.resize_bgr used_rga=%s shape=%s" % (used, out.shape))

    if failed:
        print("VALIDATION FAILED: diferencias por encima del umbral")
        return 2

    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
