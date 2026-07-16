"""Estado base de un track (sin dependencias de ReID / deep learning)."""
from __future__ import annotations

from enum import IntEnum


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class BaseTrack:
    """Ciclo de vida comun a cualquier tracklet: id global + estado + frames."""

    _next_id = 0

    def __init__(self) -> None:
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.NEW
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @classmethod
    def next_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    @classmethod
    def reset_id(cls) -> None:
        """Reinicia el contador global (tests, o reset de sesion si se decide en Fase 6)."""
        cls._next_id = 0

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED
