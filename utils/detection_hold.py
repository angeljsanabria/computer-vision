"""
Hold de detecciones/tracks entre inferencias (EVERY_N skip + miss TTL).

Contrato:
  SKIPPED  — el modelo no corrio este frame: conservar hold; no tracker.update.
  DETECTED — hay dets utiles: refrescar hold + tracker.update(dets).
  EMPTY    — el modelo corrio y no hay objeto: miss; tras max_misses, soltar hold.

Ausencia de inferencia != evidencia de ausencia.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


class InferKind(Enum):
    SKIPPED = auto()
    EMPTY = auto()
    DETECTED = auto()


@dataclass(frozen=True, slots=True)
class InferOutcome:
    """Resultado de un tick de detector (RF, YOLO, etc.)."""

    kind: InferKind
    dets: Any = None

    @staticmethod
    def skipped() -> InferOutcome:
        return InferOutcome(InferKind.SKIPPED)

    @staticmethod
    def empty() -> InferOutcome:
        return InferOutcome(InferKind.EMPTY)

    @staticmethod
    def detected(dets: Any) -> InferOutcome:
        return InferOutcome(InferKind.DETECTED, dets=dets)

    @property
    def is_detected(self) -> bool:
        return self.kind is InferKind.DETECTED

    @property
    def fresh_dets(self) -> Any | None:
        """Dets solo si hubo DETTECTED este frame (embed / FSM fresca)."""
        return self.dets if self.kind is InferKind.DETECTED else None


@dataclass(slots=True)
class DetectionHold:
    """
    Estado de overlay entre frames de inferencia.

    ``update_tracks(dets)`` — ByteTrack (o None) con dets del hit.
    ``clear_tracks()`` — reset del tracker al soltar hold (empty/None segun adapter).
    """

    dets: Any | None = None
    tracks: Any | None = None
    misses: int = 0

    @property
    def has_hold(self) -> bool:
        return self.dets is not None or self.tracks is not None

    def apply(
        self,
        outcome: InferOutcome,
        *,
        max_misses: int,
        update_tracks: Callable[[Any], Any | None],
        clear_tracks: Callable[[], None],
    ) -> None:
        if outcome.kind is InferKind.SKIPPED:
            return

        if outcome.kind is InferKind.DETECTED:
            self.dets = outcome.dets
            self.tracks = update_tracks(outcome.dets)
            self.misses = 0
            return

        # EMPTY: solo cuenta miss si hay algo que retener.
        if not self.has_hold:
            return
        self.misses += 1
        if self.misses >= max(1, int(max_misses)):
            clear_tracks()
            self.clear()

    def clear(self) -> None:
        self.dets = None
        self.tracks = None
        self.misses = 0

    def force_clear(self, clear_tracks: Callable[[], None]) -> None:
        """Gate FSM off / IDLE: soltar hold y resetear tracker si habia estado."""
        if self.has_hold:
            clear_tracks()
        self.clear()
