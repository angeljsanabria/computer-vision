"""Tipos del snapshot publico expuesto por GET /api/v1/vision-status."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class VisionPublicStatus(str, Enum):
    """Estado de negocio visible para consumidores HTTP."""

    NO_DETECCION_FACE = "NO_DETECCION_FACE"
    DETECCION_FACES = "DETECCION_FACES"
    DETECTION_AND_RECOGNIZED = "DETECTION_AND_RECOGNIZED"


def now_iso() -> str:
    """Timestamp UTC ISO 8601 para JSON (updated_at)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VisionSnapshot:
    """Ultimo estado publicado por el pipeline (inmutable)."""

    status: VisionPublicStatus
    person_id: str | None
    nombre: str | None
    face_count: int
    refresh_remaining_s: int
    updated_at: str

    @classmethod
    def no_deteccion_face(cls, *, updated_at: str | None = None) -> VisionSnapshot:
        """Estado inicial: sin sesion facial / sin caras."""
        return cls(
            status=VisionPublicStatus.NO_DETECCION_FACE,
            person_id=None,
            nombre=None,
            face_count=0,
            refresh_remaining_s=0,
            updated_at=updated_at or now_iso(),
        )

    def to_api_dict(self) -> dict:
        """Dict fijo para la respuesta JSON del endpoint."""
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "nombre": self.nombre,
            "face_count": self.face_count,
            "refresh_remaining_s": self.refresh_remaining_s,
            "updated_at": self.updated_at,
        }
