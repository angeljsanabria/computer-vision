"""
Gate de color HSV post-YOLO / pre-tracker (anti-fantasma).

Rechaza detecciones de una clase si el crop de la bbox no cumple criterios
HSV (OpenCV) sobre el frame BGR original.

Pipeline:
  detect -> verificar_color_bidones -> verificar_color_picos
         -> mejores_bidones -> hold -> ByteTrack

Bidon (class_id=0): rojo vivo del plastico (H estrecho, S/V altos; excluye bordo).
Pico (class_id=1): verde dominante, o verde + metal (sin comprobacion de rojo).
  Ratios por cuadrantes (grilla 2x2): max por celda, independiente de orientacion.
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

# Nombres fila-major para grilla 2x2 (TL, TR, BL, BR).
_CELDA_NOMBRES_2X2 = ("TL", "TR", "BL", "BR")


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


def ratio_hsv_en_rangos(
    hsv: np.ndarray,
    ranges: Sequence[HsvRange],
) -> float:
    """Fraccion de pixeles de un array HSV en la union de ``ranges``."""
    if hsv is None or hsv.size == 0 or not ranges:
        return 0.0
    total = int(hsv.shape[0] * hsv.shape[1])
    if total <= 0:
        return 0.0
    mask = cv2.inRange(hsv, ranges[0].lower_u8(), ranges[0].upper_u8())
    for rng in ranges[1:]:
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, rng.lower_u8(), rng.upper_u8())
        )
    return float(cv2.countNonZero(mask)) / float(total)


def ratio_en_rangos_hsv(
    crop_bgr: np.ndarray,
    ranges: Sequence[HsvRange],
    *,
    hsv: np.ndarray | None = None,
) -> float:
    """Fraccion de pixeles del crop que caen en la union de ``ranges``."""
    if crop_bgr is None or crop_bgr.size == 0 or not ranges:
        return 0.0
    if hsv is None:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return ratio_hsv_en_rangos(hsv, ranges)


def _nombre_celda(fila: int, col: int, filas: int, cols: int) -> str:
    if filas == 2 and cols == 2:
        return _CELDA_NOMBRES_2X2[fila * cols + col]
    return f"R{fila}C{col}"


def ratios_por_celdas_hsv(
    hsv: np.ndarray,
    ranges: Sequence[HsvRange],
    *,
    filas: int,
    cols: int,
    celda_min_px: int,
) -> list[tuple[str, float]]:
    """
    Ratio HSV por celda de una grilla filas x cols (orden fila-major).

    Si alguna celda queda por debajo de ``celda_min_px``, usa el crop completo
    como una sola celda ``full`` (bbox demasiado chica para cuadrantes).
    """
    if hsv is None or hsv.size == 0 or not ranges:
        return [("full", 0.0)]
    h, w = hsv.shape[:2]
    if filas < 1 or cols < 1:
        return [("full", ratio_hsv_en_rangos(hsv, ranges))]

    celda_h = h // filas
    celda_w = w // cols
    if celda_h * celda_w < int(celda_min_px):
        return [("full", ratio_hsv_en_rangos(hsv, ranges))]

    out: list[tuple[str, float]] = []
    for fila in range(filas):
        y0 = fila * h // filas
        y1 = (fila + 1) * h // filas if fila < filas - 1 else h
        for col in range(cols):
            x0 = col * w // cols
            x1 = (col + 1) * w // cols if col < cols - 1 else w
            sub = hsv[y0:y1, x0:x1]
            if sub.size == 0:
                out.append((_nombre_celda(fila, col, filas, cols), 0.0))
                continue
            ratio = ratio_hsv_en_rangos(sub, ranges)
            out.append((_nombre_celda(fila, col, filas, cols), ratio))
    return out


def mejor_ratio_celdas(
    hsv: np.ndarray,
    ranges: Sequence[HsvRange],
    *,
    filas: int,
    cols: int,
    celda_min_px: int,
) -> tuple[float, str]:
    """Max ratio entre celdas de la grilla; devuelve (ratio, nombre_celda_ganadora)."""
    celdas = ratios_por_celdas_hsv(
        hsv,
        ranges,
        filas=filas,
        cols=cols,
        celda_min_px=celda_min_px,
    )
    if not celdas:
        return 0.0, "?"
    nombre, ratio = max(celdas, key=lambda item: item[1])
    return float(ratio), str(nombre)


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
    log_decisions: bool = False,
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

    log = logging.info if log_decisions else logging.debug

    kept: list[np.ndarray] = []
    for row in raw.dets:
        if int(row[5]) != int(class_id):
            kept.append(row)
            continue
        crop = crop_con_inset(frame_bgr, row, inset)
        if crop is None:
            log(
                "[NozzleColor] %s rechazo: bbox invalida score=%.2f inset=%.2f",
                name,
                float(row[4]),
                float(inset),
            )
            continue
        ratio = ratio_en_rangos_hsv(crop, ranges)
        if ratio >= float(ratio_min):
            kept.append(row)
            log(
                "[NozzleColor] %s OK: score=%.2f ratio=%.3f >= min=%.3f inset=%.2f",
                name,
                float(row[4]),
                ratio,
                float(ratio_min),
                float(inset),
            )
        else:
            log(
                "[NozzleColor] %s rechazo: score=%.2f ratio=%.3f < min=%.3f inset=%.2f",
                name,
                float(row[4]),
                ratio,
                float(ratio_min),
                float(inset),
            )

    if not kept:
        return NozzleBidonDetections.empty()
    return NozzleBidonDetections(
        dets=np.stack(kept, axis=0).astype(np.float32, copy=False)
    )


def _cumple_color_pico(
    crop_bgr: np.ndarray,
    *,
    ratio_verde_min: float,
    ratio_metal_min: float,
    ratio_verde_solo_min: float,
    verde: HsvRange,
    metal: HsvRange,
    grid_filas: int,
    grid_cols: int,
    celda_min_px: int,
) -> tuple[bool, float, float, str, str]:
    """
    Valida Pico en un crop (una conversion BGR->HSV).

    Ratios verde/metal = max entre cuadrantes de la grilla (p. ej. 2x2).

    Regla: verde >= ratio_verde_solo_min, o bien
           verde >= ratio_verde_min y metal >= ratio_metal_min.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grid_kw = {
        "filas": grid_filas,
        "cols": grid_cols,
        "celda_min_px": celda_min_px,
    }
    ratio_verde, celda_verde = mejor_ratio_celdas(hsv, (verde,), **grid_kw)
    ratio_metal, celda_metal = mejor_ratio_celdas(hsv, (metal,), **grid_kw)

    if ratio_verde >= float(ratio_verde_solo_min):
        return True, ratio_verde, ratio_metal, celda_verde, celda_metal

    ok = ratio_verde >= float(ratio_verde_min) and ratio_metal >= float(
        ratio_metal_min
    )
    return ok, ratio_verde, ratio_metal, celda_verde, celda_metal


def _motivo_pico_ok(
    rv: float,
    rm: float,
    *,
    ratio_verde_min: float,
    ratio_metal_min: float,
    ratio_verde_solo_min: float,
) -> str:
    if rv >= float(ratio_verde_solo_min):
        return "verde_solo"
    if rv >= float(ratio_verde_min) and rm >= float(ratio_metal_min):
        return "verde+metal"
    return "?"


def filtrar_clase_pico_por_color(
    frame_bgr: np.ndarray,
    raw: NozzleBidonDetections,
    *,
    inset: float,
    ratio_verde_min: float,
    ratio_metal_min: float,
    ratio_verde_solo_min: float,
    verde: HsvRange,
    metal: HsvRange,
    grid_filas: int,
    grid_cols: int,
    celda_min_px: int,
    log_decisions: bool = False,
) -> NozzleBidonDetections:
    """Rechaza filas Pico (class_id=1) que no cumplen criterio verde / verde+metal HSV."""
    if not raw.has_detections:
        return raw

    log = logging.info if log_decisions else logging.debug

    kept: list[np.ndarray] = []
    for row in raw.dets:
        if int(row[5]) != CLASS_ID_PICO:
            kept.append(row)
            continue
        crop = crop_con_inset(frame_bgr, row, inset)
        if crop is None:
            log(
                "[NozzleColor] Pico rechazo: bbox invalida score=%.2f inset=%.2f",
                float(row[4]),
                float(inset),
            )
            continue
        ok, rv, rm, cv, cm = _cumple_color_pico(
            crop,
            ratio_verde_min=ratio_verde_min,
            ratio_metal_min=ratio_metal_min,
            ratio_verde_solo_min=ratio_verde_solo_min,
            verde=verde,
            metal=metal,
            grid_filas=grid_filas,
            grid_cols=grid_cols,
            celda_min_px=celda_min_px,
        )
        if ok:
            kept.append(row)
            log(
                "[NozzleColor] Pico OK: score=%.2f via=%s "
                "verde=%.3f@%s metal=%.3f@%s inset=%.2f grid=%dx%d "
                "(min verde=%.3f metal=%.3f verde_solo=%.3f)",
                float(row[4]),
                _motivo_pico_ok(
                    rv,
                    rm,
                    ratio_verde_min=ratio_verde_min,
                    ratio_metal_min=ratio_metal_min,
                    ratio_verde_solo_min=ratio_verde_solo_min,
                ),
                rv,
                cv,
                rm,
                cm,
                float(inset),
                int(grid_filas),
                int(grid_cols),
                float(ratio_verde_min),
                float(ratio_metal_min),
                float(ratio_verde_solo_min),
            )
        else:
            log(
                "[NozzleColor] Pico rechazo: score=%.2f "
                "verde=%.3f@%s metal=%.3f@%s inset=%.2f grid=%dx%d "
                "(min verde=%.3f metal=%.3f verde_solo=%.3f)",
                float(row[4]),
                rv,
                cv,
                rm,
                cm,
                float(inset),
                int(grid_filas),
                int(grid_cols),
                float(ratio_verde_min),
                float(ratio_metal_min),
                float(ratio_verde_solo_min),
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

    ``ranges`` suele ser los dos lobulos del rojo OpenCV.
    """
    if not enabled:
        return raw
    n_candidatos = sum(
        1 for row in raw.dets if int(row[5]) == CLASS_ID_BIDON
    )
    out = filtrar_clase_por_color(
        frame_bgr,
        raw,
        class_id=CLASS_ID_BIDON,
        ratio_min=ratio_min,
        inset=inset,
        ranges=ranges,
        label="Bidon",
        log_decisions=True,
    )
    n_ok = (
        sum(1 for row in out.dets if int(row[5]) == CLASS_ID_BIDON)
        if out.has_detections
        else 0
    )
    if n_candidatos > 0:
        logging.info(
            "[NozzleColor] Bidon resumen: %d candidato(s) YOLO -> %d pasaron color "
            "(ratio_min=%.3f inset=%.2f)",
            n_candidatos,
            n_ok,
            float(ratio_min),
            float(inset),
        )
    return out


def verificar_color_picos(
    frame_bgr: np.ndarray,
    raw: NozzleBidonDetections,
    *,
    enabled: bool,
    inset: float,
    ratio_verde_min: float,
    ratio_metal_min: float,
    ratio_verde_solo_min: float,
    verde: HsvRange,
    metal: HsvRange,
    grid_filas: int = 2,
    grid_cols: int = 2,
    celda_min_px: int = 64,
) -> NozzleBidonDetections:
    """
    Gate anti-fantasma Pico. Si ``enabled`` es False, devuelve ``raw``.

    ``verde``: cuerpo plastico (HSV amplio, stream/luz). ``metal``: cuello plateado.
    Ratios por max en cuadrantes ``grid_filas`` x ``grid_cols`` (TL/TR/BL/BR si 2x2).
    """
    if not enabled:
        return raw
    n_candidatos = sum(
        1 for row in raw.dets if int(row[5]) == CLASS_ID_PICO
    )
    out = filtrar_clase_pico_por_color(
        frame_bgr,
        raw,
        inset=inset,
        ratio_verde_min=ratio_verde_min,
        ratio_metal_min=ratio_metal_min,
        ratio_verde_solo_min=ratio_verde_solo_min,
        verde=verde,
        metal=metal,
        grid_filas=grid_filas,
        grid_cols=grid_cols,
        celda_min_px=celda_min_px,
        log_decisions=True,
    )
    n_ok = (
        sum(1 for row in out.dets if int(row[5]) == CLASS_ID_PICO)
        if out.has_detections
        else 0
    )
    if n_candidatos > 0:
        logging.info(
            "[NozzleColor] Pico resumen: %d candidato(s) YOLO -> %d pasaron color "
            "(verde_min=%.3f metal_min=%.3f verde_solo_min=%.3f "
            "inset=%.2f grid=%dx%d celda_min_px=%d)",
            n_candidatos,
            n_ok,
            float(ratio_verde_min),
            float(ratio_metal_min),
            float(ratio_verde_solo_min),
            float(inset),
            int(grid_filas),
            int(grid_cols),
            int(celda_min_px),
        )
    return out
