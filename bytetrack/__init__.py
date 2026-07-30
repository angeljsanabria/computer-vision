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


def build_nozzle_tracker(
    enable: bool,
    config: ByteTrackConfig | None = None,
    *,
    show_bbox_score: float = 0.0,
    show_bbox_hits: int = 1,
) -> "NozzleByteTracker | None":
    """Tracker ByteTrack para detecciones nozzle (siempre ON si ENABLE_NOZZLE)."""
    if not enable:
        return None
    from bytetrack.nozzle_tracker import NozzleByteTracker

    return NozzleByteTracker(
        config or ByteTrackConfig(),
        show_bbox_score=show_bbox_score,
        show_bbox_hits=show_bbox_hits,
    )


__all__ = [
    "ByteTrackConfig",
    "FaceTrack",
    "FaceTracker",
    "TrackResult",
    "build_face_tracker",
    "build_nozzle_tracker",
]
