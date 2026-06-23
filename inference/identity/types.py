"""Tipos de salida del matcher de identidad."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityMatch:
    """Resultado 1:N: mejor identidad de la galeria y si supera el umbral coseno."""

    label: str
    similarity: float
    is_match: bool
