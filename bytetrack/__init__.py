"""Tracking visual de caras (ByteTrack). Sin acoplamiento a embed/identidad."""
from __future__ import annotations

from bytetrack.types import (
    ByteTrackConfig,
    FaceTrack,
    FaceTracker,
    TrackResult,
)


def build_face_tracker(
    enable: bool, config: ByteTrackConfig | None = None
) -> FaceTracker | None:
    """Factory (mismo patron que build_face_detector / build_embedder en inference/)."""
    if not enable:
        return None
    from bytetrack.face_tracker import FaceByteTracker

    return FaceByteTracker(config or ByteTrackConfig())


__all__ = [
    "ByteTrackConfig",
    "FaceTrack",
    "FaceTracker",
    "TrackResult",
    "build_face_tracker",
]
