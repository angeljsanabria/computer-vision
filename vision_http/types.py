"""Tipos del snapshot publico expuesto por GET /api/v1/vision-status."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class VisionPublicStatus(str, Enum):
    """Estado de negocio visible para consumidores HTTP."""

    NO_FACE_DETECTION = "NO_FACE_DETECTION"
    FACES_DETECTED = "FACES_DETECTED"
    FACE_RECOGNIZED = "FACE_RECOGNIZED"


def now_iso() -> str:
    """Timestamp UTC ISO 8601 para JSON (updated_at)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VisionSnapshot:
    """Ultimo estado publicado por el pipeline (inmutable)."""

    status: VisionPublicStatus
    person_id: str | None
    name: str | None
    face_count: int
    refresh_remaining_s: int
    updated_at: str

    @classmethod
    def no_face_detection(cls, *, updated_at: str | None = None) -> VisionSnapshot:
        """Estado inicial: sin sesion facial / sin caras."""
        return cls(
            status=VisionPublicStatus.NO_FACE_DETECTION,
            person_id=None,
            name=None,
            face_count=0,
            refresh_remaining_s=0,
            updated_at=updated_at or now_iso(),
        )

    def to_api_dict(self) -> dict:
        """Dict fijo para la respuesta JSON del endpoint."""
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "name": self.name,
            "face_count": self.face_count,
            "refresh_remaining_s": self.refresh_remaining_s,
            "updated_at": self.updated_at,
        }
