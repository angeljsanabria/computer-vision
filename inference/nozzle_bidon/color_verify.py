"""
Gate de color HSV post-YOLO / pre-tracker (anti-fantasma).

Rechaza detecciones de una clase si el crop de la bbox no alcanza
un ratio minimo de pixeles dentro de uno o mas rangos HSV (OpenCV).

Pipeline:
  detect -> verificar_color_* -> mejores_bidones -> hold -> ByteTrack

Bidon (class_id=0): espectro rojo (dos rangos por wrap-around de H).
Pico: mismo mecanismo con otro espectro cuando se habilite.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from inference.nozzle_bidon.constants import CLASS_NAMES
from inference.nozzle_bidon.types import NozzleBidonDetections

CLASS_ID_BIDON = 0
CLASS_ID_PICO = 1


@dataclass(frozen=True)
class HsvRange:
    """Un rango HSV inclusivo (OpenCV: H 0-180, S/V 0-255)."""

    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int

    def lower_u8(self) -> np.ndarray:
        return np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)

    def upper_u8(self) -> np.ndarray:
        return np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)


def ratio_en_rangos_hsv(crop_bgr: np.ndarray, ranges: Sequence[HsvRange]) -> float:
    """Fraccion de pixeles del crop que caen en la union de ``ranges``."""
    if crop_bgr is None or crop_bgr.size == 0 or not ranges:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ranges[0].lower_u8(), ranges[0].upper_u8())
    for rng in ranges[1:]:
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, rng.lower_u8(), rng.upper_u8())
        )
    total = int(crop_bgr.shape[0] * crop_bgr.shape[1])
    if total <= 0:
        return 0.0
    return float(cv2.countNonZero(mask)) / float(total)


def crop_con_inset(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    inset: float,
) -> np.ndarray | None:
    """Crop BGR de xyxy con inset fraccionario hacia adentro (menos fondo)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 2.0 or bh < 2.0:
        return None
    dx = bw * float(inset)
    dy = bh * float(inset)
    ix1 = int(max(0, min(w - 1, round(x1 + dx))))
    iy1 = int(max(0, min(h - 1, round(y1 + dy))))
    ix2 = int(max(0, min(w, round(x2 - dx))))
    iy2 = int(max(0, min(h, round(y2 - dy))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return frame_bgr[iy1:iy2, ix1:ix2]


def filtrar_clase_por_color(
    frame_bgr: np.ndarray,
    raw: NozzleBidonDetections,
    *,
    class_id: int,
    ratio_min: float,
    inset: float,
    ranges: Sequence[HsvRange],
    label: str | None = None,
) -> NozzleBidonDetections:
    """
    Rechaza filas de ``class_id`` cuyo crop no cumple ``ratio_min`` en ``ranges``.
    Otras clases pasan sin cambio.
    """
    if not raw.has_detections:
        return raw

    if label is not None:
        name = label
    elif 0 <= class_id < len(CLASS_NAMES):
        name = CLASS_NAMES[class_id]
    else:
        name = str(class_id)

    kept: list[np.ndarray] = []
    for row in raw.dets:
        if int(row[5]) != int(class_id):
            kept.append(row)
            continue
        crop = crop_con_inset(frame_bgr, row, inset)
        if crop is None:
            logging.debug(
                "[NozzleColor] %s rechazo: bbox invalida score=%.2f",
                name,
                float(row[4]),
            )
            continue
        ratio = ratio_en_rangos_hsv(crop, ranges)
        if ratio >= float(ratio_min):
            kept.append(row)
        else:
            logging.debug(
                "[NozzleColor] %s rechazo: score=%.2f ratio=%.3f < min=%.3f",
                name,
                float(row[4]),
                ratio,
                float(ratio_min),
            )

    if not kept:
        return NozzleBidonDetections.empty()
    return NozzleBidonDetections(
        dets=np.stack(kept, axis=0).astype(np.float32, copy=False)
    )


def verificar_color_bidones(
    frame_bgr: np.ndarray,
    raw: NozzleBidonDetections,
    *,
    enabled: bool,
    ratio_min: float,
    inset: float,
    ranges: Sequence[HsvRange],
) -> NozzleBidonDetections:
    """
    Gate anti-fantasma Bidon. Si ``enabled`` es False, devuelve ``raw``.

    ``ranges`` suele ser los dos lobulos del rojo OpenCV. Pico usara el mismo
    helper ``filtrar_clase_por_color`` con CLASS_ID_PICO y su propio espectro.
    """
    if not enabled:
        return raw
    return filtrar_clase_por_color(
        frame_bgr,
        raw,
        class_id=CLASS_ID_BIDON,
        ratio_min=ratio_min,
        inset=inset,
        ranges=ranges,
        label="Bidon",
    )
