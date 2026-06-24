"""Matching coseno live vs galeria (L2-normalizada en enrolamiento)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference.identity.types import IdentityMatch
from inference.mobilefacenet.constants import EMBED_DIM

GALLERY_NPY_NAME = "gallery.npy"
GALLERY_META_NAME = "gallery_meta.json"
DATA_NORMALIZADA_KEY = "data_normalizada"
DATA_NORMALIZADA_OK = 1


@dataclass(frozen=True)
class _GalleryLoad:
    refs: tuple[tuple[str, np.ndarray], ...]
    matrix: np.ndarray | None


class FaceGalleryMatcher:
    """
    Galeria 1:N: ``gallery.npy`` + ``gallery_meta.json`` (preferido) o un .npy
    por identidad (legacy).

    Referencias y vector live deben venir ya L2-normalizados. El JSON debe incluir
    ``data_normalizada: 1`` (sin recalcular norma en placa). Similitud = producto punto.
    """

    def __init__(
        self,
        gallery_dir: Path,
        min_similarity: float,
    ) -> None:
        self._min_similarity = float(min_similarity)
        loaded = self._load_gallery(gallery_dir)
        self._refs = loaded.refs
        self._matrix = loaded.matrix
        if self._refs:
            labels = ", ".join(label for label, _ in self._refs)
            logging.info(
                "Galeria identidad: %d refs en %s (%s)",
                len(self._refs),
                gallery_dir,
                labels,
            )
        else:
            logging.warning(
                "Galeria identidad vacia o inexistente: %s (sin MATCH posible)",
                gallery_dir,
            )

    @classmethod
    def from_settings(cls) -> FaceGalleryMatcher:
        from configs import settings as s

        return cls(
            gallery_dir=Path(s.embed_ref_gallery_dir_path()),
            min_similarity=s.EMBED_SIM_MIN_MATCH,
        )

    @staticmethod
    def _load_gallery(gallery_dir: Path) -> _GalleryLoad:
        if not gallery_dir.is_dir():
            return _GalleryLoad(refs=(), matrix=None)

        matrix_load = FaceGalleryMatcher._try_load_gallery_matrix(gallery_dir)
        if matrix_load is not None:
            return matrix_load

        refs = FaceGalleryMatcher._load_gallery_legacy(gallery_dir)
        return _GalleryLoad(refs=tuple(refs), matrix=None)

    @staticmethod
    def _try_load_gallery_matrix(gallery_dir: Path) -> _GalleryLoad | None:
        npy_path = gallery_dir / GALLERY_NPY_NAME
        meta_path = gallery_dir / GALLERY_META_NAME
        if not npy_path.is_file() and not meta_path.is_file():
            return None
        if not npy_path.is_file() or not meta_path.is_file():
            raise ValueError(
                f"Galeria incompleta en {gallery_dir}: requiere "
                f"{GALLERY_NPY_NAME} y {GALLERY_META_NAME} juntos"
            )

        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        if meta.get(DATA_NORMALIZADA_KEY) != DATA_NORMALIZADA_OK:
            raise ValueError(
                f"{GALLERY_META_NAME} invalido: falta {DATA_NORMALIZADA_KEY}="
                f"{DATA_NORMALIZADA_OK} (got {meta.get(DATA_NORMALIZADA_KEY)!r}). "
                "Regenerar con embeddings/face_embeddings_npy_from_images_folder.py"
            )

        gallery = np.load(str(npy_path)).astype(np.float32, copy=False)
        if gallery.ndim != 2 or gallery.shape[1] != EMBED_DIM:
            raise ValueError(
                f"{GALLERY_NPY_NAME} shape invalida {gallery.shape!r}, "
                f"esperado (N, {EMBED_DIM})"
            )

        entries = meta.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"{GALLERY_META_NAME}: entries debe ser un array JSON")
        if len(entries) != gallery.shape[0]:
            raise ValueError(
                f"{GALLERY_META_NAME}: len(entries)={len(entries)} != "
                f"filas gallery={gallery.shape[0]}"
            )

        refs: list[tuple[str, np.ndarray]] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{GALLERY_META_NAME}: entries[{i}] no es objeto")
            person_id = entry.get("id")
            if not person_id:
                raise ValueError(f"{GALLERY_META_NAME}: entries[{i}] sin id")
            refs.append((str(person_id), gallery[i]))

        return _GalleryLoad(refs=tuple(refs), matrix=gallery)

    @staticmethod
    def _load_gallery_legacy(gallery_dir: Path) -> list[tuple[str, np.ndarray]]:
        refs: list[tuple[str, np.ndarray]] = []
        for path in sorted(gallery_dir.glob("*.npy")):
            if path.name == GALLERY_NPY_NAME:
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
            refs.append((path.stem, vec))

        return refs

    @property
    def count(self) -> int:
        return len(self._refs)

    def match(self, vector: np.ndarray) -> IdentityMatch | None:
        """
        Devuelve la identidad con mayor similitud y si alcanza ``min_similarity``.

        ``vector`` debe ser el retorno de ``embedder.embed()``: float32, shape
        (EMBED_DIM,), ya L2-normalizado. No se re-normaliza aqui.

        ``None`` si la galeria esta vacia o el vector live no tiene dimension valida.
        """
        if not self._refs:
            return None

        live = vector.reshape(-1)
        if live.size != EMBED_DIM:
            logging.warning(
                "Match: vector live size=%d, esperado %d; omitiendo",
                live.size,
                EMBED_DIM,
            )
            return None

        if self._matrix is not None:
            sims = self._matrix @ live
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])
            best_label = self._refs[best_i][0]
        else:
            best_label = self._refs[0][0]
            best_sim = -1.0
            for label, ref in self._refs:
                sim = float(np.dot(ref, live))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        return IdentityMatch(
            label=best_label,
            similarity=best_sim,
            is_match=best_sim >= self._min_similarity,
        )
