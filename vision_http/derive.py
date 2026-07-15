"""Derivacion del estado publico HTTP desde el snapshot interno del pipeline."""
from __future__ import annotations

from typing import Any

from .types import VisionPublicStatus, VisionSnapshot, now_iso


def _face_count(dets: Any | None) -> int:
    if dets is None or not dets.has_faces:
        return 0
    return int(dets.dets.shape[0])


def _refresh_to_int(refresh_remaining_s: float | None) -> int:
    if refresh_remaining_s is None:
        return 0
    return max(0, int(refresh_remaining_s))


def derive_vision_status(
    *,
    fsm_state: Any,
    dets: Any | None,
    display_identity: Any | None,
    refresh_remaining_s: float | None,
) -> VisionSnapshot:
    """Construye el estado publico sin modificar la FSM ni el pipeline."""
    face_count = _face_count(dets)
    state_value = getattr(fsm_state, "value", fsm_state)
    refresh_int = _refresh_to_int(refresh_remaining_s)

    # Retencion FSM: FACE_RECOGNIZED gana aunque este frame no haya dets (cooldown).
    if state_value == "FACE_RECOGNIZED" and display_identity is not None:
        return VisionSnapshot(
            status=VisionPublicStatus.FACE_RECOGNIZED,
            person_id=display_identity.person_id,
            name=display_identity.nombre,
            face_count=face_count,
            refresh_remaining_s=refresh_int,
            updated_at=now_iso(),
        )

    if state_value == "IDLE" or face_count == 0:
        return VisionSnapshot.no_face_detection()

    return VisionSnapshot(
        status=VisionPublicStatus.FACES_DETECTED,
        person_id=None,
        name=None,
        face_count=face_count,
        refresh_remaining_s=0,
        updated_at=now_iso(),
    )
