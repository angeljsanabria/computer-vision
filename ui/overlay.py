"""Anotaciones de depuracion sobre frames BGR."""
from __future__ import annotations

import cv2
import numpy as np

from inference.nozzle_bidon.types import NozzleBidonDetections
from inference.types import FaceDetections
from ui.facemesh_overlay import draw_facemesh_points
from ui.overlay_text import OverlayTextRenderer
from ui.overlay_theme import NOZZLE_BBOX_BGR, OverlayTheme
from ui.types import FrameView


# class_id 0 = Bidon (inference.nozzle_bidon.constants.CLASS_NAMES)
_BIDON_CLASS_ID = 0
_WARNING_OBJECT_MSG = "Se identifico un objeto no autorizado"
# Naranja alerta BGR (distinto del match verde/azul).
_WARNING_BAR_BGR = (0, 140, 255)


def _has_visible_bidon(view: FrameView) -> bool:
    """True si hay Bidon dibujable (track show_bbox o det cruda de fallback)."""
    if view.nozzle_tracks is not None:
        for track in view.nozzle_tracks.tracks:
            if track.show_bbox and track.class_id == _BIDON_CLASS_ID:
                return True
        return False
    dets = view.nozzle_dets
    if dets is None or not dets.has_detections:
        return False
    for row in dets.dets:
        if int(row[5]) == _BIDON_CLASS_ID:
            return True
    return False


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Color BGR estable por track_id (misma idea que el bytetrack/visualize.py original)."""
    r = (track_id * 43) % 256
    g = (track_id * 97) % 256
    b = (track_id * 113) % 256
    return (b, g, r)


class DebugOverlay:
    """Dibuja bbox/tracks e identidad sobre una copia del frame."""

    def __init__(
        self,
        theme: OverlayTheme | None = None,
        *,
        show_identity_bar: bool = True,
        warning_object_bar: bool = False,
    ) -> None:
        self._theme = theme if theme is not None else OverlayTheme.legacy()
        self._text = OverlayTextRenderer(self._theme)
        self._show_identity_bar = show_identity_bar
        self._warning_object_bar = warning_object_bar

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
        self._draw_nozzle(vis, view)
        self._draw_bottom_bar(vis, view)
        return vis

    def _draw_faces(self, vis: np.ndarray, dets: FaceDetections | None) -> None:
        if dets is None or not dets.has_faces:
            return
        for idx, row in enumerate(dets.dets):
            x1, y1, x2, y2 = map(int, row[:4])
            score = float(row[4])
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            self._text.draw_label(vis, x1, y1, f"#{idx + 1} {score:.2f}", color)

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
                color = self._theme.match_color(is_stale=view.identity_is_stale)
            else:
                label = f"Desconocido\n# {track.track_id}"
                color = _track_color(track.track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # FaceMesh UX: solo desconocidos (MATCH gana aunque queden landmarks viejos).
            if not is_match:
                landmarks = mesh_map.get(track.track_id)
                if landmarks is not None:
                    draw_facemesh_points(vis, landmarks, point_color=color)
            self._text.draw_label(vis, x1, y1, label, color)

    def _draw_nozzle(self, vis: np.ndarray, view: FrameView) -> None:
        """
        Bboxes nozzle: prioriza tracks con ``show_bbox`` (sticky). Sin texto de clase
        (evita mostrar Bidon/Pico en falsos positivos).

        Si no hay ningun track mostrable (tracks vacios, score bajo el umbral
        de display, o id nuevo sin hits), fallback a dets del hold para no
        perder el bbox mientras el tracker re-confirma.
        """
        drew_track = False
        if view.nozzle_tracks is not None:
            color = NOZZLE_BBOX_BGR
            for track in view.nozzle_tracks.tracks:
                if not track.show_bbox:
                    continue
                x1, y1, x2, y2 = map(int, track.tlbr)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                drew_track = True
        if drew_track:
            return
        if view.nozzle_dets is not None and view.nozzle_dets.has_detections:
            self._draw_nozzle_dets(vis, view.nozzle_dets)

    def _draw_nozzle_dets(
        self,
        vis: np.ndarray,
        dets: NozzleBidonDetections | None,
    ) -> None:
        if dets is None or not dets.has_detections:
            return
        color = NOZZLE_BBOX_BGR
        for row in dets.dets:
            x1, y1, x2, y2 = map(int, row[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    def _draw_bottom_bar(self, vis: np.ndarray, view: FrameView) -> None:
        """
        Barra inferior UX: aviso Bidon y/o identidad MATCH.

        El aviso usa el mismo strip que la barra de identidad; si hay Bidon
        visible (show_bbox) tiene prioridad de color (alerta).
        """
        warn = self._warning_object_bar and _has_visible_bidon(view)
        idm = view.identity
        show_id = (
            self._show_identity_bar
            and idm is not None
            and idm.is_match
        )
        if not warn and not show_id:
            return

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 32), (w, h), (0, 0, 0), -1)

        if warn and show_id:
            bar = f"{_WARNING_OBJECT_MSG}  |  {idm.nombre}  ID: {idm.person_id}"
            color = _WARNING_BAR_BGR
        elif warn:
            bar = _WARNING_OBJECT_MSG
            color = _WARNING_BAR_BGR
        else:
            bar = f"{idm.nombre}  ID: {idm.person_id}"
            color = self._theme.match_color(is_stale=view.identity_is_stale)

        self._text.draw_bar_text(vis, bar, x=6, baseline_y=h - 10, color=color)
