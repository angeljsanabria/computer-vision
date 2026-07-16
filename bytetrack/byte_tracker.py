"""Nucleo ByteTrack: IoU + Kalman en dos pasos, sin YOLOX/torch/escala de letterbox."""
from __future__ import annotations

import numpy as np

from bytetrack import matching
from bytetrack.basetrack import BaseTrack, TrackState
from bytetrack.kalman_filter import KalmanFilter
from bytetrack.types import ByteTrackConfig

_LOW_SCORE_FLOOR = 0.1
_SECOND_ASSOC_THRESH = 0.5
_UNCONFIRMED_THRESH = 0.7
_DUPLICATE_IOU_THRESH = 0.15


class STrack(BaseTrack):
    """Un tracklet: bbox tlwh + filtro de Kalman propio."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float) -> None:
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.score = float(score)
        self.tracklet_len = 0

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list["STrack"]) -> None:
        if not stracks:
            return
        means = np.asarray([t.mean.copy() for t in stracks])
        covariances = np.asarray([t.covariance for t in stracks])
        for i, t in enumerate(stracks):
            if t.state != TrackState.TRACKED:
                means[i][7] = 0
        means, covariances = STrack.shared_kalman.multi_predict(means, covariances)
        for t, mean, cov in zip(stracks, means, covariances):
            t.mean = mean
            t.covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Arranca un tracklet nuevo (deteccion sin match)."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Actualiza un track ya asociado con su nueva deteccion."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self) -> np.ndarray:
        """Posicion actual top-left (x, y, w, h). Usa Kalman si ya fue activado."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """(x1, y1, x2, y2) para IoU y overlay."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"STrack_{self.track_id}({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    Asociacion en dos pasos (alta confianza + baja confianza) mas Kalman.

    Entrada de ``update``: array (N, 5) = [x1, y1, x2, y2, score] en pixeles del
    frame original (mismo sistema de coordenadas que FaceDetections). Sin
    escalado de letterbox: el adapter ya entrega coords de frame.

    Con RetinaFace filtrando por score alto antes de llegar aqui, el paso de
    "baja confianza" suele quedar vacio; se mantiene por fidelidad al algoritmo.
    """

    def __init__(self, config: ByteTrackConfig) -> None:
        self._cfg = config
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.det_thresh = config.track_thresh + 0.1
        self.max_time_lost = int(config.frame_rate / 30.0 * config.track_buffer)
        self.kalman_filter = KalmanFilter()

    def update(self, dets_xyxy_score: np.ndarray) -> list[STrack]:
        self.frame_id += 1
        activated: list[STrack] = []
        refound: list[STrack] = []
        lost: list[STrack] = []
        removed: list[STrack] = []

        dets = np.asarray(dets_xyxy_score, dtype=np.float32).reshape(-1, 5)
        scores = dets[:, 4]
        bboxes = dets[:, :4]

        high_mask = scores >= self._cfg.track_thresh
        low_mask = (scores >= _LOW_SCORE_FLOOR) & ~high_mask

        detections = self._make_stracks(bboxes[high_mask], scores[high_mask])
        detections_second = self._make_stracks(bboxes[low_mask], scores[low_mask])

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        strack_pool = _joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = matching.fuse_score(matching.iou_distance(strack_pool, detections), detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self._cfg.match_thresh
        )
        for i_track, i_det in matches:
            track, det = strack_pool[i_track], detections[i_det]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refound.append(track)

        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.TRACKED]
        dists = matching.iou_distance(r_tracked, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=_SECOND_ASSOC_THRESH)
        for i_track, i_det in matches:
            track, det = r_tracked[i_track], detections_second[i_det]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refound.append(track)

        for i in u_track:
            track = r_tracked[i]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost.append(track)

        remaining_dets = [detections[i] for i in u_detection]
        dists = matching.fuse_score(
            matching.iou_distance(unconfirmed, remaining_dets), remaining_dets
        )
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=_UNCONFIRMED_THRESH
        )
        for i_track, i_det in matches:
            unconfirmed[i_track].update(remaining_dets[i_det], self.frame_id)
            activated.append(unconfirmed[i_track])
        for i in u_unconfirmed:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        for i in u_detection:
            track = remaining_dets[i]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refound)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _make_stracks(bboxes: np.ndarray, scores: np.ndarray) -> list[STrack]:
        if bboxes.shape[0] == 0:
            return []
        return [STrack(STrack.tlbr_to_tlwh(box), score) for box, score in zip(bboxes, scores)]


def _joint_stracks(list_a: list[STrack], list_b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in list_a}
    result = list(list_a)
    for t in list_b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            result.append(t)
    return result


def _sub_stracks(list_a: list[STrack], list_b: list[STrack]) -> list[STrack]:
    exclude = {t.track_id for t in list_b}
    return [t for t in list_a if t.track_id not in exclude]


def _remove_duplicate_stracks(
    stracks_a: list[STrack], stracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    if not stracks_a or not stracks_b:
        return stracks_a, stracks_b
    pdist = matching.iou_distance(stracks_a, stracks_b)
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(pdist < _DUPLICATE_IOU_THRESH)):
        age_a = stracks_a[p].frame_id - stracks_a[p].start_frame
        age_b = stracks_b[q].frame_id - stracks_b[q].start_frame
        (dup_b if age_a > age_b else dup_a).add(q if age_a > age_b else p)
    result_a = [t for i, t in enumerate(stracks_a) if i not in dup_a]
    result_b = [t for i, t in enumerate(stracks_b) if i not in dup_b]
    return result_a, result_b
