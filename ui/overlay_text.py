"""Dibujo de labels de overlay: Hershey (legacy) o TrueType via Pillow (CTK)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import ImageFont

    from ui.overlay_theme import OverlayTheme


class OverlayTextRenderer:
    """
    Render de texto sobre frames BGR.

    Carga la fuente TrueType una sola vez (cache por tamaño en px).
    Si el tema no usa TTF, delega a ``cv2.putText`` (mismo aspecto historico).
    """

    def __init__(self, theme: OverlayTheme) -> None:
        self._theme = theme
        self._font_by_px: dict[int, ImageFont.FreeTypeFont] = {}
        self._use_truetype = bool(theme.use_truetype and theme.font_path is not None)
        if self._use_truetype:
            self._warm_fonts()

    def _warm_fonts(self) -> None:
        try:
            self._font(self._theme.label_font_px)
            self._font(self._theme.bar_font_px)
        except Exception as exc:
            logging.info(
                "CTK font: no se pudo cargar %s (%s); se usa Hershey.",
                self._theme.font_path,
                exc,
            )
            self._use_truetype = False
            self._font_by_px.clear()

    def _font(self, size_px: int):
        from PIL import ImageFont

        cached = self._font_by_px.get(size_px)
        if cached is not None:
            return cached
        path = self._theme.font_path
        assert path is not None
        font = ImageFont.truetype(str(path), size_px)
        self._font_by_px[size_px] = font
        return font

    def draw_label(
        self,
        vis: np.ndarray,
        x1: int,
        y1: int,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        """Label con fondo solido. ``\\n`` separa lineas."""
        if self._use_truetype:
            self._draw_label_ttf(vis, x1, y1, label, color)
        else:
            self._draw_label_hershey(vis, x1, y1, label, color)

    def draw_bar_text(
        self,
        vis: np.ndarray,
        text: str,
        *,
        x: int,
        baseline_y: int,
        color: tuple[int, int, int],
    ) -> None:
        if self._use_truetype:
            self._draw_bar_ttf(vis, text, x=x, baseline_y=baseline_y, color=color)
        else:
            cv2.putText(
                vis,
                text,
                (x, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_label_hershey(
        self,
        vis: np.ndarray,
        x1: int,
        y1: int,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        lines = label.split("\n")
        sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for line in lines
        ]
        line_h = sizes[0][1] + 10
        tw = max(w for w, _ in sizes)
        txt_y1 = max(y1 - line_h * len(lines), 0)
        cv2.rectangle(
            vis, (x1, txt_y1), (x1 + tw, txt_y1 + line_h * len(lines)), color, -1
        )
        for i, line in enumerate(lines):
            cv2.putText(
                vis,
                line,
                (x1, txt_y1 + line_h * (i + 1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._theme.label_text_bgr,
                1,
                cv2.LINE_AA,
            )

    def _draw_label_ttf(
        self,
        vis: np.ndarray,
        x1: int,
        y1: int,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        from PIL import Image, ImageDraw

        lines = label.split("\n")
        font = self._font(self._theme.label_font_px)
        pad_x, pad_y, gap = 6, 4, 2

        widths: list[int] = []
        heights: list[int] = []
        for line in lines:
            box = font.getbbox(line)
            widths.append(max(0, box[2] - box[0]))
            heights.append(max(0, box[3] - box[1]))

        content_w = max(widths) if widths else 0
        content_h = sum(heights) + gap * max(0, len(lines) - 1)
        plate_w = content_w + 2 * pad_x
        plate_h = content_h + 2 * pad_y
        if plate_w <= 0 or plate_h <= 0:
            return

        txt_y1 = max(y1 - plate_h, 0)
        frame_h, frame_w = vis.shape[:2]
        plate_w = min(plate_w, max(1, frame_w - x1))
        plate_h = min(plate_h, max(1, frame_h - txt_y1))

        rgb = (
            self._theme.label_text_bgr[2],
            self._theme.label_text_bgr[1],
            self._theme.label_text_bgr[0],
        )
        bg = (color[2], color[1], color[0])
        img = Image.new("RGB", (plate_w, plate_h), bg)
        draw = ImageDraw.Draw(img)
        y = pad_y
        for line, lh in zip(lines, heights):
            draw.text((pad_x, y), line, font=font, fill=rgb)
            y += lh + gap

        patch = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        vis[txt_y1 : txt_y1 + plate_h, x1 : x1 + plate_w] = patch

    def _draw_bar_ttf(
        self,
        vis: np.ndarray,
        text: str,
        *,
        x: int,
        baseline_y: int,
        color: tuple[int, int, int],
    ) -> None:
        from PIL import Image, ImageDraw

        font = self._font(self._theme.bar_font_px)
        box = font.getbbox(text)
        tw = max(1, box[2] - box[0] + 4)
        th = max(1, box[3] - box[1] + 4)
        top = max(0, baseline_y - th + 2)
        frame_h, frame_w = vis.shape[:2]
        tw = min(tw, max(1, frame_w - x))
        th = min(th, max(1, frame_h - top))

        roi = vis[top : top + th, x : x + tw]
        if roi.size == 0:
            return
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        rgb = (color[2], color[1], color[0])
        draw.text((2, 0), text, font=font, fill=rgb)
        vis[top : top + th, x : x + tw] = cv2.cvtColor(
            np.asarray(img), cv2.COLOR_RGB2BGR
        )
