"""Adapter: NozzleBidonDetections -> TrackResult (IDs estables, overlay)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bytetrack.byte_tracker import BYTETracker, STrack
from bytetrack.matching import ious
from bytetrack.types import ByteTrackConfig, FaceTrack, TrackResult

if TYPE_CHECKING:
    from inference.nozzle_bidon.types import NozzleBidonDetections

# IoU minimo para asociar class_id de una det al track (cache de etiqueta UI).
_CLASS_MATCH_IOU_MIN = 0.3


class NozzleByteTracker:
    """
    ByteTrack sobre detecciones Bidon/Pico (independiente del tracker facial).

    Lee ``NozzleBidonDetections`` sin mutarlas (xyxy+score; class_id no entra
    a ByteTrack). UI:
      - ``show_bbox`` sticky tras N hits con score >= umbral
      - ``class_id`` cacheado por IoU con dets del frame (ultima clase conocida)
    """

    def __init__(
        self,
        config: ByteTrackConfig,
        *,
        show_bbox_score: float = 0.0,
        show_bbox_hits: int = 1,
    ) -> None:
        self._tracker = BYTETracker(config)
        self._show_bbox_score = float(show_bbox_score)
        self._show_bbox_hits = max(1, int(show_bbox_hits))
        self._track_buffer = max(1, int(config.track_buffer))
        self._show_bbox_by_id: dict[int, bool] = {}
        self._score_hits_by_id: dict[int, int] = {}
        self._class_id_by_id: dict[int, int] = {}
        self._absent_frames: dict[int, int] = {}

    def update(self, dets: NozzleBidonDetections | None) -> TrackResult:
        stracks = self._tracker.update(_to_tracker_input(dets))
        active_ids = {int(t.track_id) for t in stracks}
        self._prune_absent(active_ids)

        if not stracks:
            return TrackResult.empty()

        matched_class = _match_class_ids(stracks, dets)
        tracks: list[FaceTrack] = []
        for strack in stracks:
            tid = int(strack.track_id)
            self._absent_frames[tid] = 0

            if tid in matched_class:
                self._class_id_by_id[tid] = matched_class[tid]
            class_id = self._class_id_by_id.get(tid)

            show = self._update_show_bbox(tid, float(strack.score))
            tracks.append(
                FaceTrack(
                    track_id=tid,
                    tlbr=strack.tlbr.astype(np.float32, copy=True),
                    score=float(strack.score),
                    show_bbox=show,
                    class_id=class_id,
                )
            )
        return TrackResult(tracks=tuple(tracks))

    def _update_show_bbox(self, track_id: int, score: float) -> bool:
        """Sticky: True permanente tras ``show_bbox_hits`` frames consecutivos OK."""
        if self._show_bbox_by_id.get(track_id, False):
            return True

        if score >= self._show_bbox_score:
            hits = self._score_hits_by_id.get(track_id, 0) + 1
        else:
            hits = 0
        self._score_hits_by_id[track_id] = hits

        if hits >= self._show_bbox_hits:
            self._show_bbox_by_id[track_id] = True
            return True
        return False

    def _prune_absent(self, active_ids: set[int]) -> None:
        """Libera estado UI de tracks muertos (respeta buffer de ByteTrack)."""
        known = set(self._show_bbox_by_id) | set(self._score_hits_by_id) | set(
            self._class_id_by_id
        )
        for tid in known:
            if tid in active_ids:
                continue
            n = self._absent_frames.get(tid, 0) + 1
            self._absent_frames[tid] = n
            if n > self._track_buffer:
                self._show_bbox_by_id.pop(tid, None)
                self._score_hits_by_id.pop(tid, None)
                self._class_id_by_id.pop(tid, None)
                self._absent_frames.pop(tid, None)


def _to_tracker_input(dets: NozzleBidonDetections | None) -> np.ndarray:
    if dets is None or not dets.has_detections:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(dets.dets[:, :5], dtype=np.float32, copy=True)


def _match_class_ids(
    stracks: list[STrack],
    dets: NozzleBidonDetections | None,
) -> dict[int, int]:
    """Asigna class_id por mejor IoU (greedy) entre track y dets del frame."""
    if dets is None or not dets.has_detections or not stracks:
        return {}

    track_boxes = np.stack(
        [t.tlbr.astype(np.float32, copy=False) for t in stracks], axis=0
    )
    det_boxes = np.asarray(dets.dets[:, :4], dtype=np.float32)
    det_cls = dets.dets[:, 5]
    iou_mat = ious(track_boxes, det_boxes)

    # Greedy: pares (track_i, det_j) por IoU descendente, sin reusar det.
    pairs: list[tuple[float, int, int]] = []
    for i in range(iou_mat.shape[0]):
        for j in range(iou_mat.shape[1]):
            iou = float(iou_mat[i, j])
            if iou >= _CLASS_MATCH_IOU_MIN:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    out: dict[int, int] = {}
    for _iou, i, j in pairs:
        if i in used_tracks or j in used_dets:
            continue
        used_tracks.add(i)
        used_dets.add(j)
        out[int(stracks[i].track_id)] = int(det_cls[j])
    return out
