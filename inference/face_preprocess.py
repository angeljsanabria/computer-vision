"""
Preproceso cara 112x112 para MobileFaceNet.

Prioridad (mutuamente excluyente en uso; si ambos true gana ArcFace):
  1. ``FACE_ALIGNMENT_ENABLE`` — siempre align ArcFace 5 puntos (galeria alineada).
  2. ``FACE_ROT_ALIGNMENT_SIMPLE_ENABLE`` — hibrido crop / roll-fix si |roll| > umbral.
  3. Ninguno — solo crop bbox + resize (export_models).
"""
from __future__ import annotations

from dataclasses import dataclass

from inference.face_align import MOBILEFACENET_ALIGN_SIZE, align_face_from_det_row
from inference.face_crop import bbox_crop_with_margin, crop_bbox_to_size
from inference.face_pose import eye_roll_deg_from_det_row, roll_within_frontal_range
from inference.face_roll_fix import crop_roll_fix_to_size

__all__ = [
    "FacePatch",
    "bbox_crop_with_margin",
    "crop_bbox_to_size",
    "prepare_face_patch",
    "prepare_face_patch_from_settings",
]


@dataclass(frozen=True)
class FacePatch:
    """Parche BGR 112x112 listo para el embedder."""

    bgr: np.ndarray
    used_roll_fix: bool
    used_arcface_align: bool
    roll_deg: float


def prepare_face_patch(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
    *,
    arcface_align_enable: bool,
    rot_align_simple_enable: bool,
    max_abs_roll_deg: float,
    crop_margin_frac: float,
    out_size: int = MOBILEFACENET_ALIGN_SIZE,
) -> FacePatch:
    """
    Genera parche BGR para embedding.

    arcface_align_enable: siempre warp ArcFace (requiere refs enroladas igual).
    rot_align_simple_enable: crop si |roll| <= umbral; si no roll-fix en el crop.
    Ambos false: solo crop bbox + resize.
    """
    if arcface_align_enable:
        roll_deg = eye_roll_deg_from_det_row(det_row)
        bgr = align_face_from_det_row(frame_bgr, det_row, image_size=out_size)
        return FacePatch(
            bgr=bgr,
            used_roll_fix=False,
            used_arcface_align=True,
            roll_deg=roll_deg,
        )

    if not rot_align_simple_enable:
        bgr = crop_bbox_to_size(
            frame_bgr,
            det_row,
            margin_frac=crop_margin_frac,
            out_size=out_size,
        )
        return FacePatch(
            bgr=bgr,
            used_roll_fix=False,
            used_arcface_align=False,
            roll_deg=0.0,
        )

    roll_deg = eye_roll_deg_from_det_row(det_row)

    if roll_within_frontal_range(roll_deg, max_abs_roll_deg):
        bgr = crop_bbox_to_size(
            frame_bgr,
            det_row,
            margin_frac=crop_margin_frac,
            out_size=out_size,
        )
        return FacePatch(
            bgr=bgr,
            used_roll_fix=False,
            used_arcface_align=False,
            roll_deg=roll_deg,
        )

    bgr, _ = crop_roll_fix_to_size(
        frame_bgr,
        det_row,
        margin_frac=crop_margin_frac,
        out_size=out_size,
    )
    return FacePatch(
        bgr=bgr,
        used_roll_fix=True,
        used_arcface_align=False,
        roll_deg=roll_deg,
    )


def prepare_face_patch_from_settings(
    frame_bgr: np.ndarray,
    det_row: np.ndarray,
) -> FacePatch:
    """Atajo: lee flags y umbrales de ``configs.settings``."""
    from configs import settings as s

    return prepare_face_patch(
        frame_bgr,
        det_row,
        arcface_align_enable=s.FACE_ALIGNMENT_ENABLE,
        rot_align_simple_enable=s.FACE_ROT_ALIGNMENT_SIMPLE_ENABLE,
        max_abs_roll_deg=s.FACE_ROLL_MAX_DEG,
        crop_margin_frac=s.FACE_CROP_MARGIN_FRAC,
    )
