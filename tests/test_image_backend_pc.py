"""Tests de image backend en PC (OpenCV exclusivo, sin RGA)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.image_backend import (
    effective_use_rga,
    opencv_letterbox_bgr,
    opencv_resize,
    resize_bgr,
    should_use_rga,
)
from utils.image_utils import LetterboxMeta, bgr_to_rgb, letterbox_bgr, resize_frame


@pytest.fixture(autouse=True)
def _pc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "pc")
    monkeypatch.setenv("USE_RGA", "false")


def test_should_use_rga_false_on_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "pc")
    monkeypatch.setenv("USE_RGA", "true")
    assert should_use_rga() is False
    assert effective_use_rga(explicit=True) is False


def test_should_use_rga_only_rk3568(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "rk3568")
    monkeypatch.setenv("USE_RGA", "true")
    assert should_use_rga() is True
    assert effective_use_rga() is True


def test_resize_frame_matches_opencv() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    out_wh = (320, 240)
    expected = opencv_resize(frame, out_wh, cv2.INTER_AREA)
    got = resize_frame(frame, out_wh, interpolation=cv2.INTER_AREA)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_letterbox_bgr_meta_and_shape() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    canvas, meta = letterbox_bgr(frame, (320, 320), fill_value=114)
    assert canvas.shape == (320, 320, 3)
    assert isinstance(meta, LetterboxMeta)
    exp_canvas, exp_ar, exp_ox, exp_oy = opencv_letterbox_bgr(frame, (320, 320), 114)
    assert meta.aspect_ratio == pytest.approx(exp_ar)
    assert meta.offset_x == exp_ox
    assert meta.offset_y == exp_oy
    np.testing.assert_array_equal(canvas, exp_canvas)


def test_bgr_to_rgb_channel_order() -> None:
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    assert rgb.shape == (1, 1, 3)
    assert tuple(rgb[0, 0]) == (30, 20, 10)


def test_resize_bgr_ignores_use_rga_on_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "pc")
    monkeypatch.setenv("USE_RGA", "true")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = resize_bgr(frame, (50, 50), cv2.INTER_LINEAR, use_rga=True)
    expected = cv2.resize(frame, (50, 50), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(out, expected)
