"""Adapter: NozzleDetections -> TrackResult (IDs estables, solo overlay)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bytetrack.byte_tracker import BYTETracker, STrack
from bytetrack.types import ByteTrackConfig, FaceTrack, TrackResult

if TYPE_CHECKING:
    from inference.nozzle.types import NozzleDetections


class NozzleByteTracker:
    """
    ByteTrack sobre detecciones nozzle (independiente del tracker facial).

    Lee ``NozzleDetections`` sin mutarla.
    """

    def __init__(self, config: ByteTrackConfig) -> None:
        self._tracker = BYTETracker(config)

    def update(self, dets: NozzleDetections | None) -> TrackResult:
        stracks = self._tracker.update(_to_tracker_input(dets))
        if not stracks:
            return TrackResult.empty()
        return TrackResult(tracks=tuple(_to_face_track(t) for t in stracks))


def _to_tracker_input(dets: NozzleDetections | None) -> np.ndarray:
    if dets is None or not dets.has_detections:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(dets.dets[:, :5], dtype=np.float32, copy=True)


def _to_face_track(strack: STrack) -> FaceTrack:
    return FaceTrack(
        track_id=strack.track_id,
        tlbr=strack.tlbr.astype(np.float32, copy=True),
        score=strack.score,
    )
