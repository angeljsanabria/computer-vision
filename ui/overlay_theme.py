"""Tema visual del overlay (legacy debug vs branding CTK)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# Azul CTK: #0547C5 (RGB 5,71,197) -> BGR OpenCV
# Nozzle Bidon/Pico bbox: #0449CE (RGB 4,73,206) -> BGR OpenCV
# Verde Autorizado #267C19 (RGB 38,124,25) -> BGR OpenCV
_CTK_MATCH_BGR: tuple[int, int, int] = (197, 71, 5)
NOZZLE_BBOX_BGR: tuple[int, int, int] = (206, 73, 4)
_VERDE_MATCH_BGR: tuple[int, int, int] = (25, 124, 38)
_MATCH_BBOX_SELECTED: tuple[int, int, int] = _VERDE_MATCH_BGR
_LEGACY_MATCH_BGR: tuple[int, int, int] = (0, 200, 0)
_STALE_BGR: tuple[int, int, int] = (0, 0, 255)
# Texto sobre placa de label bbox (caras): #000000
_LABEL_TEXT_BGR: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class OverlayTheme:
    """
    Paleta y tipografia del overlay.

    ``use_truetype=False`` -> cv2.FONT_HERSHEY_SIMPLEX (comportamiento historico).
    """

    match_bgr: tuple[int, int, int]
    match_stale_bgr: tuple[int, int, int]
    label_text_bgr: tuple[int, int, int]
    use_truetype: bool
    font_path: Path | None
    label_font_px: int
    bar_font_px: int

    @classmethod
    def legacy(cls) -> OverlayTheme:
        return cls(
            match_bgr=_LEGACY_MATCH_BGR,
            match_stale_bgr=_STALE_BGR,
            label_text_bgr=_LABEL_TEXT_BGR,
            use_truetype=False,
            font_path=None,
            label_font_px=14,
            bar_font_px=16,
        )

    @classmethod
    def ctk(cls, *, font_path: Path | None, use_truetype: bool) -> OverlayTheme:
        return cls(
            match_bgr=_MATCH_BBOX_SELECTED,
            match_stale_bgr=_STALE_BGR,
            label_text_bgr=_LABEL_TEXT_BGR,
            use_truetype=use_truetype,
            font_path=font_path if use_truetype else None,
            label_font_px=15,
            bar_font_px=17,
        )

    def match_color(self, *, is_stale: bool) -> tuple[int, int, int]:
        return self.match_stale_bgr if is_stale else self.match_bgr


def resolve_overlay_theme(
    *,
    ctk_colors_and_font: bool,
    font_path: str,
) -> OverlayTheme:
    """
    Construye el tema al arranque.

    El path de fuente se interpreta igual que ``DISPLAY_BANNER_PATH``
    (relativo al cwd / valor de env). Si no existe el archivo: colores CTK + Hershey.
    """
    if not ctk_colors_and_font:
        return OverlayTheme.legacy()

    path = Path(font_path) if font_path else None
    if path is None or not path.is_file():
        logging.info(
            "CTK_COLORS_AND_FONT=true: no se encontro la fuente '%s'; "
            "colores CTK con Hershey.",
            font_path,
        )
        return OverlayTheme.ctk(font_path=None, use_truetype=False)

    logging.info("CTK overlay: color match #0547C5 + fuente %s", path)
    return OverlayTheme.ctk(font_path=path, use_truetype=True)
