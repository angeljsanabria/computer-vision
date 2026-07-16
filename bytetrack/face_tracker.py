"""Adapter: FaceDetections (RetinaFace) -> TrackResult (IDs estables, solo overlay)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bytetrack.byte_tracker import BYTETracker, STrack
from bytetrack.types import ByteTrackConfig, FaceTrack, TrackResult

if TYPE_CHECKING:
    from inference.types import FaceDetections


class FaceByteTracker:
    """
    Implementa el Protocol ``FaceTracker`` (bytetrack.types).

    Lee ``FaceDetections`` sin mutarla: copia ``dets[:, :5]`` (bbox + score;
    landmarks ignorados) antes de pasarla al nucleo ByteTrack. El array
    original de RetinaFace sigue intacto para embed / matcher / FSM.
    """

    def __init__(self, config: ByteTrackConfig) -> None:
        self._tracker = BYTETracker(config)

    def update(self, dets: FaceDetections | None) -> TrackResult:
        stracks = self._tracker.update(_to_tracker_input(dets))
        if not stracks:
            return TrackResult.empty()
        return TrackResult(tracks=tuple(_to_face_track(t) for t in stracks))


def _to_tracker_input(dets: FaceDetections | None) -> np.ndarray:
    """Copia defensiva bbox+score; nunca escribe sobre ``dets.dets``."""
    if dets is None or not dets.has_faces:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(dets.dets[:, :5], dtype=np.float32, copy=True)


def _to_face_track(strack: STrack) -> FaceTrack:
    return FaceTrack(
        track_id=strack.track_id,
        tlbr=strack.tlbr.astype(np.float32, copy=True),
        score=strack.score,
    )
