"""Anotaciones de depuracion sobre frames BGR."""
from __future__ import annotations

import cv2
import numpy as np

from inference.types import FaceDetections
from ui.facemesh_overlay import draw_facemesh_points
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
    """Dibuja bbox/tracks e identidad sobre una copia del frame."""

    def __init__(self) -> None:
        self._keep_alive_phase = 0

    def render(self, frame_bgr: np.ndarray, view: FrameView) -> np.ndarray:
        vis = frame_bgr.copy()
        if view.tracks is not None:
            # Tracking activo: solo se dibujan tracks confirmados por ByteTrack.
            # Detecciones sin track asociado (aun sin confirmar, o por debajo de
            # det_thresh) no se muestran; evita mezclar el estilo "#n score" de
            # RetinaFace con el de tracking en la misma pantalla.
            self._draw_tracks(vis, view)
        else:
            # Tracking desactivado (ENABLE_FACE_TRACKING=false): dets crudos de RetinaFace.
            self._draw_faces(vis, view.dets)
        self._draw_identity(vis, view)
        return vis

    def draw_keep_alive(self, vis: np.ndarray, *, below_y: int = 0) -> None:
        """
        Indicador de vida sobre el canvas final.

        ``below_y``: borde inferior del banner. Si es > 0, dibuja justo debajo
        a la derecha; si es 0, esquina inferior derecha.
        """
        h, w = vis.shape[:2]
        if h <= 0 or w <= 0:
            return

        alive = self._next_keep_alive()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2
        (tw, th), _ = cv2.getTextSize(alive, font, scale, thickness)
        pad = 4
        box_w = tw + pad * 2
        box_h = th + pad * 2

        if below_y > 0:
            # Pegado al borde inferior del banner (derecha).
            x1 = max(0, w - box_w - 8)
            y1 = min(below_y + 1, max(0, h - box_h))
        else:
            x1 = max(0, w - box_w - 8)
            y1 = max(0, h - box_h - 8)

        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            vis,
            alive,
            (x1 + pad, y2 - pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _next_keep_alive(self) -> str:
        dots = "." * (self._keep_alive_phase + 1)
        self._keep_alive_phase = (self._keep_alive_phase + 1) % 3
        return dots

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
        mesh_map = view.facemesh_by_track or {}
        for track in view.tracks.tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            idm = id_map.get(track.track_id)
            if idm is None and view.identity is not None and track.track_id == view.identity_track_id:
                idm = view.identity
            is_match = idm is not None and idm.is_match
            if is_match:
                label = f"{idm.nombre}\nID: {idm.person_id}"
                color = (0, 0, 255) if view.identity_is_stale else (0, 200, 0)
            else:
                label = f"Desconocido\n# {track.track_id}"
                color = _track_color(track.track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # FaceMesh UX: solo desconocidos (MATCH gana aunque queden landmarks viejos).
            if not is_match:
                landmarks = mesh_map.get(track.track_id)
                if landmarks is not None:
                    draw_facemesh_points(vis, landmarks, point_color=color)
            _draw_label(vis, x1, y1, label, color)

    def _draw_identity(self, vis: np.ndarray, view: FrameView) -> None:
        """Barra inferior solo con identidad confirmada (activa o retenida)."""
        idm = view.identity
        if idm is None or not idm.is_match:
            return

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 32), (w, h), (0, 0, 0), -1)

        bar = f"{idm.nombre}  ID: {idm.person_id}"
        color = (0, 0, 255) if view.identity_is_stale else (0, 200, 0)

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
