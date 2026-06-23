"""Matching coseno live vs galeria de embeddings .npy (L2-normalizados)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.identity.types import IdentityMatch
from inference.mobilefacenet.constants import EMBED_DIM
from inference.mobilefacenet.norm import l2_normalize


class FaceGalleryMatcher:
    """
    Galeria 1:N en disco: un .npy por identidad (nombre del archivo = label).

    Similitud: producto punto entre vectores unitarios (coseno), igual que
    export_models/RetinaFace_from_cam_with_id.py.
    """

    def __init__(
        self,
        gallery_dir: Path,
        min_similarity: float,
    ) -> None:
        self._min_similarity = float(min_similarity)
        self._refs: tuple[tuple[str, np.ndarray], ...] = tuple(
            self._load_gallery(gallery_dir)
        )
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
    def _load_gallery(gallery_dir: Path) -> list[tuple[str, np.ndarray]]:
        if not gallery_dir.is_dir():
            return []

        refs: list[tuple[str, np.ndarray]] = []
        for path in sorted(gallery_dir.glob("*.npy")):
            vec = np.load(str(path)).astype(np.float32).reshape(-1)
            if vec.size != EMBED_DIM:
                logging.warning(
                    "Galeria: omitiendo %s (size=%d, esperado %d)",
                    path.name,
                    vec.size,
                    EMBED_DIM,
                )
                continue
            refs.append((path.stem, l2_normalize(vec)))

        return refs

    @property
    def count(self) -> int:
        return len(self._refs)

    def match(self, vector: np.ndarray) -> IdentityMatch | None:
        """
        Devuelve la identidad con mayor similitud y si alcanza ``min_similarity``.

        ``None`` solo si la galeria no tiene referencias validas.
        """
        if not self._refs:
            return None

        live = l2_normalize(vector)
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
