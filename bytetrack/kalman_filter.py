"""Filtro de Kalman para bounding boxes (solo NumPy, sin scipy)."""
from __future__ import annotations

import numpy as np

_NDIM = 4
_DT = 1.0


class KalmanFilter:
    """
    Estado (x, y, a, h, vx, vy, va, vh): centro, aspect ratio, alto y velocidades.

    Modelo de velocidad constante; observacion directa de (x, y, a, h).
    """

    def __init__(self) -> None:
        self._motion_mat = np.eye(2 * _NDIM, 2 * _NDIM)
        for i in range(_NDIM):
            self._motion_mat[i, _NDIM + i] = _DT
        self._update_mat = np.eye(_NDIM, 2 * _NDIM)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T) + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predice varios tracks a la vez.

        Loop simple (no vectorizado con einsum): con FACE_PROCESS_TOP_N tipico
        (1-2 caras) el costo es despreciable y se prioriza claridad.
        """
        out_means = np.zeros_like(means)
        out_covs = np.zeros_like(covariances)
        for i in range(len(means)):
            out_means[i], out_covs[i] = self.predict(means[i], covariances[i])
        return out_means, out_covs

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        kalman_gain = np.linalg.solve(
            projected_cov, (covariance @ self._update_mat.T).T
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance
