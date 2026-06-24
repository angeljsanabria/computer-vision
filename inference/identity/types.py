"""Tipos de salida del matcher de identidad."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GalleryEntry:
    """Metadatos de una fila de gallery.npy (misma fila en gallery_meta.json)."""

    person_id: str
    nombre: str
    rotacion: str


@dataclass(frozen=True)
class IdentityMatch:
    """Resultado 1:N: mejor fila de la galeria y si supera el umbral coseno."""

    person_id: str
    nombre: str
    rotacion: str
    row_index: int
    similarity: float
    is_match: bool

    @property
    def label(self) -> str:
        """Alias legacy: id de persona (0000000001)."""
        return self.person_id
