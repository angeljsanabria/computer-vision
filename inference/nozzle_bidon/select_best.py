"""Ranking de detecciones Bidon/Pico (top-N por score)."""
from __future__ import annotations

import numpy as np

from inference.nozzle_bidon.types import NozzleBidonDetections


def mejores_bidones(
    raw: NozzleBidonDetections,
    *,
    top_n: int,
    min_score: float,
) -> NozzleBidonDetections:
    """Filtra por score minimo y conserva las ``top_n`` mejores."""
    if not raw.has_detections:
        return NozzleBidonDetections.empty()
    rows = raw.dets[raw.dets[:, 4] >= float(min_score)]
    if rows.shape[0] == 0:
        return NozzleBidonDetections.empty()
    order = np.argsort(-rows[:, 4])
    limit = max(1, int(top_n))
    return NozzleBidonDetections(dets=rows[order[:limit]].astype(np.float32, copy=False))
