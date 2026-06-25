"""Matching coseno live vs galeria (L2-normalizada en enrolamiento)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference.identity.types import GalleryEntry, IdentityMatch
from inference.mobilefacenet.constants import EMBED_DIM

GALLERY_NPY_NAME = "gallery.npy"
GALLERY_META_NAME = "gallery_meta.json"
GALLERY_ALIGN_NPY_NAME = "gallery_align.npy"
GALLERY_ALIGN_META_NAME = "gallery_meta_align.json"
DATA_NORMALIZADA_KEY = "data_normalizada"
DATA_NORMALIZADA_OK = 1


@dataclass(frozen=True)
class _GalleryLoad:
    entries: tuple[GalleryEntry, ...]
    matrix: np.ndarray | None


class FaceGalleryMatcher:
    """
    Galeria 1:N: ``gallery.npy`` + ``gallery_meta.json`` (preferido) o un .npy
    por identidad (legacy).

    Referencias y vector live deben venir ya L2-normalizados. El JSON debe incluir
    ``data_normalizada: 1`` (sin recalcular norma en placa). Similitud = producto punto
    via ``gallery @ live`` -> ``argmax`` vs ``min_similarity``.
    """

    def __init__(
        self,
        gallery_dir: Path,
        min_similarity: float,
        npy_name: str = GALLERY_NPY_NAME,
        meta_name: str = GALLERY_META_NAME,
    ) -> None:
        self._min_similarity = float(min_similarity)
        loaded = self._load_gallery(gallery_dir, npy_name, meta_name)
        self._entries = loaded.entries
        self._matrix = loaded.matrix
        if self._entries:
            if self._matrix is not None:
                n, d = self._matrix.shape
                logging.info(
                    "Galeria identidad: %d refs matriz (%d, %d) en %s | sim_min=%.2f",
                    len(self._entries),
                    n,
                    d,
                    gallery_dir,
                    self._min_similarity,
                )
            else:
                labels = ", ".join(e.person_id for e in self._entries)
                logging.info(
                    "Galeria identidad: %d refs legacy en %s (%s) | sim_min=%.2f",
                    len(self._entries),
                    gallery_dir,
                    labels,
                    self._min_similarity,
                )
        else:
            logging.warning(
                "Galeria identidad vacia o inexistente: %s (sin MATCH posible)",
                gallery_dir,
            )

    @classmethod
    def from_settings(cls) -> FaceGalleryMatcher:
        from configs import settings as s

        if s.FACE_ALIGNMENT_ENABLE:
            npy_name = GALLERY_ALIGN_NPY_NAME
            meta_name = GALLERY_ALIGN_META_NAME
        else:
            npy_name = GALLERY_NPY_NAME
            meta_name = GALLERY_META_NAME

        return cls(
            gallery_dir=Path(s.embed_ref_gallery_dir_path()),
            min_similarity=s.EMBED_SIM_MIN_MATCH,
            npy_name=npy_name,
            meta_name=meta_name,
        )

    @staticmethod
    def _load_gallery(
        gallery_dir: Path, npy_name: str, meta_name: str
    ) -> _GalleryLoad:
        if not gallery_dir.is_dir():
            return _GalleryLoad(entries=(), matrix=None)

        matrix_load = FaceGalleryMatcher._try_load_gallery_matrix(
            gallery_dir, npy_name, meta_name
        )
        if matrix_load is not None:
            return matrix_load

        return FaceGalleryMatcher._load_gallery_legacy(gallery_dir)

    @staticmethod
    def _parse_entry(entry: object, index: int, meta_name: str) -> GalleryEntry:
        if not isinstance(entry, dict):
            raise ValueError(f"{meta_name}: entries[{index}] no es objeto")
        person_id = entry.get("id")
        if not person_id:
            raise ValueError(f"{meta_name}: entries[{index}] sin id")
        nombre = str(entry.get("nombre") or person_id)
        rotacion = str(entry.get("rotacion") or "")
        return GalleryEntry(
            person_id=str(person_id),
            nombre=nombre,
            rotacion=rotacion,
        )

    @staticmethod
    def _try_load_gallery_matrix(
        gallery_dir: Path, npy_name: str, meta_name: str
    ) -> _GalleryLoad | None:
        npy_path = gallery_dir / npy_name
        meta_path = gallery_dir / meta_name
        if not npy_path.is_file() and not meta_path.is_file():
            return None
        if not npy_path.is_file() or not meta_path.is_file():
            raise ValueError(
                f"Galeria incompleta en {gallery_dir}: requiere "
                f"{npy_name} y {meta_name} juntos"
            )

        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        if meta.get(DATA_NORMALIZADA_KEY) != DATA_NORMALIZADA_OK:
            raise ValueError(
                f"{meta_name} invalido: falta {DATA_NORMALIZADA_KEY}="
                f"{DATA_NORMALIZADA_OK} (got {meta.get(DATA_NORMALIZADA_KEY)!r}). "
                "Regenerar con embeddings/enroll_gallery.py"
            )

        gallery = np.load(str(npy_path)).astype(np.float32, copy=False)
        gallery = np.ascontiguousarray(gallery)
        if gallery.ndim != 2 or gallery.shape[1] != EMBED_DIM:
            raise ValueError(
                f"{npy_name} shape invalida {gallery.shape!r}, "
                f"esperado (N, {EMBED_DIM})"
            )

        entries_raw = meta.get("entries")
        if not isinstance(entries_raw, list):
            raise ValueError(f"{meta_name}: entries debe ser un array JSON")
        if len(entries_raw) != gallery.shape[0]:
            raise ValueError(
                f"{meta_name}: len(entries)={len(entries_raw)} != "
                f"filas gallery={gallery.shape[0]}"
            )

        entries = tuple(
            FaceGalleryMatcher._parse_entry(entry, i, meta_name)
            for i, entry in enumerate(entries_raw)
        )
        return _GalleryLoad(entries=entries, matrix=gallery)

    @staticmethod
    def _load_gallery_legacy(gallery_dir: Path) -> _GalleryLoad:
        entries: list[GalleryEntry] = []
        vectors: list[np.ndarray] = []
        for path in sorted(gallery_dir.glob("*.npy")):
            if path.name in (GALLERY_NPY_NAME, GALLERY_ALIGN_NPY_NAME):
                continue
            vec = np.load(str(path)).astype(np.float32).reshape(-1)
            if vec.size != EMBED_DIM:
                logging.warning(
                    "Galeria: omitiendo %s (size=%d, esperado %d)",
                    path.name,
                    vec.size,
                    EMBED_DIM,
                )
                continue
            stem = path.stem
            entries.append(
                GalleryEntry(person_id=stem, nombre=stem, rotacion="")
            )
            vectors.append(vec)

        if not entries:
            return _GalleryLoad(entries=(), matrix=None)

        matrix = np.ascontiguousarray(np.stack(vectors, axis=0))
        return _GalleryLoad(entries=tuple(entries), matrix=matrix)

    @property
    def count(self) -> int:
        return len(self._entries)

    def match(self, vector: np.ndarray) -> IdentityMatch | None:
        """
        Similitudes: ``sims = gallery @ live`` (N,). Mejor fila = ``argmax(sims)``.

        ``vector`` debe ser float32 L2-normalizado, shape (EMBED_DIM,).
        ``is_match`` es True si ``sims[best_i] >= min_similarity``.
        Metadatos (id, nombre, rotacion) vienen de ``entries[best_i]``.
        """
        if not self._entries or self._matrix is None:
            return None

        live = np.ascontiguousarray(vector.reshape(-1), dtype=np.float32)
        if live.size != EMBED_DIM:
            logging.warning(
                "Match: vector live size=%d, esperado %d; omitiendo",
                live.size,
                EMBED_DIM,
            )
            return None

        sims = self._matrix @ live
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        entry = self._entries[best_i]

        return IdentityMatch(
            person_id=entry.person_id,
            nombre=entry.nombre,
            rotacion=entry.rotacion,
            row_index=best_i,
            similarity=best_sim,
            is_match=best_sim >= self._min_similarity,
        )
