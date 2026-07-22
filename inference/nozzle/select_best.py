"""Ranking de detecciones nozzle (top-N por score)."""
from __future__ import annotations

import numpy as np

from inference.nozzle.types import NozzleDetections


def mejores_nozzles(
    raw: NozzleDetections,
    *,
    top_n: int,
    min_score: float,
) -> NozzleDetections:
    """Filtra por score minimo y conserva las ``top_n`` mejores."""
    if not raw.has_detections:
        return NozzleDetections.empty()
    rows = raw.dets[raw.dets[:, 4] >= float(min_score)]
    if rows.shape[0] == 0:
        return NozzleDetections.empty()
    order = np.argsort(-rows[:, 4])
    limit = max(1, int(top_n))
    return NozzleDetections(dets=rows[order[:limit]].astype(np.float32, copy=False))
