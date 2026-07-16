"""Anotaciones de depuracion sobre frames BGR."""
from __future__ import annotations

import cv2
import numpy as np

from inference.types import FaceDetections
from ui.types import FrameView


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Color BGR estable por track_id (misma idea que el bytetrack/visualize.py original)."""
    r = (track_id * 43) % 256
    g = (track_id * 97) % 256
    b = (track_id * 113) % 256
    return (b, g, r)


def _draw_label(
    vis: np.ndarray, x1: int, y1: int, label: str, color: tuple[int, int, int]
) -> None:
    """Label con fondo solido (legible sobre cualquier imagen de fondo). ``\\n`` separa lineas."""
    lines = label.split("\n")
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for line in lines]
    line_h = sizes[0][1] + 10
    tw = max(w for w, _ in sizes)
    txt_y1 = max(y1 - line_h * len(lines), 0)
    cv2.rectangle(vis, (x1, txt_y1), (x1 + tw, txt_y1 + line_h * len(lines)), color, -1)
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (x1, txt_y1 + line_h * (i + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


class DebugOverlay:
    """Dibuja estado FSM, MOG2 y bbox de caras (o tracks) sobre una copia del frame."""

    def __init__(self) -> None:
        self._keep_alive_phase = 0

    def render(self, frame_bgr: np.ndarray, view: FrameView) -> np.ndarray:
        vis = frame_bgr.copy()
        self._draw_keep_alive(vis)
        if view.tracks is not None:
            # Tracking activo: solo se dibujan tracks confirmados por ByteTrack.
            # Detecciones sin track asociado (aun sin confirmar, o por debajo de
            # det_thresh) no se muestran; evita mezclar el estilo "#n score" de
            # RetinaFace con el de tracking en la misma pantalla.
            self._draw_tracks(vis, view)
        else:
            # Tracking desactivado (ENABLE_FACE_TRACKING=false): dets crudos de RetinaFace.
            self._draw_faces(vis, view.dets)
        self._draw_status(vis, view)
        return vis

    def _next_keep_alive(self) -> str:
        dots = "." * (self._keep_alive_phase + 1)
        self._keep_alive_phase = (self._keep_alive_phase + 1) % 3
        return dots

    def _draw_keep_alive(self, vis: np.ndarray) -> None:
        _, w = vis.shape[:2]
        alive = self._next_keep_alive()
        (tw, th), _ = cv2.getTextSize(alive, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            vis,
            alive,
            (w - tw - 8, th + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

    def _draw_faces(self, vis: np.ndarray, dets: FaceDetections | None) -> None:
        if dets is None or not dets.has_faces:
            return
        for idx, row in enumerate(dets.dets):
            x1, y1, x2, y2 = map(int, row[:4])
            score = float(row[4])
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _draw_label(vis, x1, y1, f"#{idx + 1} {score:.2f}", color)

    def _draw_tracks(self, vis: np.ndarray, view: FrameView) -> None:
        id_map = view.identity_by_track or {}
        for track in view.tracks.tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            idm = id_map.get(track.track_id)
            if idm is None and view.identity is not None and track.track_id == view.identity_track_id:
                idm = view.identity
            if idm is not None and idm.is_match:
                label = f"{idm.nombre}\nid={idm.person_id}"
                color = (0, 0, 255) if view.identity_is_stale else (0, 200, 0)
            else:
                label = f"Desconocido\n# {track.track_id}"
                color = _track_color(track.track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _draw_label(vis, x1, y1, label, color)

    def _draw_status(self, vis: np.ndarray, view: FrameView) -> None:
        cv2.putText(
            vis,
            f"estado: {view.fsm.state.value}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"MOG2 px={view.mov.pixel_count} mov={view.mov.hay_mov}",
            (8, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        self._draw_identity(vis, view)

    def _draw_identity(self, vis: np.ndarray, view: FrameView) -> None:
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 32), (w, h), (0, 0, 0), -1)

        if view.identity is None:
            cv2.putText(
                vis,
                "Sin identidad",
                (6, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            return

        idm = view.identity
        rot = f" rot={idm.rotacion}" if idm.rotacion else ""

        if view.identity_is_stale:
            bar = (
                f"ultima: {idm.nombre} id={idm.person_id}{rot} "
                f"sim={idm.similarity:.3f} MATCH"
            )
            color = (0, 0, 255)
        elif idm.is_match:
            bar = (
                f"{idm.nombre} id={idm.person_id}{rot} "
                f"sim={idm.similarity:.3f} MATCH"
            )
            color = (0, 200, 0)
        else:
            bar = (
                f"{idm.nombre} id={idm.person_id}{rot} "
                f"sim={idm.similarity:.3f} NO_MATCH"
            )
            color = (0, 140, 255)

        cv2.putText(
            vis,
            bar,
            (6, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
