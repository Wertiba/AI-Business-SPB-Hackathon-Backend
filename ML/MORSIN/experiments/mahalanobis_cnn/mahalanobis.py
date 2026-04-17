"""Ledoit–Wolf covariance + Mahalanobis distance scoring."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisScorer:
    def __init__(self) -> None:
        self.location_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisScorer":
        lw = LedoitWolf().fit(X)
        self.location_ = lw.location_.astype(np.float64)
        cov = lw.covariance_.astype(np.float64)
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisScorer is not fitted")
        delta = X.astype(np.float64) - self.location_
        assert self.precision_ is not None
        quad = np.sum(delta @ self.precision_ * delta, axis=1)
        return np.sqrt(np.maximum(quad, 0.0))


def recall_at_fpr_threshold(
    scores_normal: np.ndarray,
    scores_positive: np.ndarray,
    *,
    fpr: float = 0.05,
) -> tuple[float, float]:
    """
    Pick threshold t = (1 - fpr) quantile of normal scores (higher score = more anomalous).
    Returns (threshold, recall_on_positives).
    """
    t = float(np.quantile(scores_normal, 1.0 - fpr))
    recall = float(np.mean(scores_positive > t))
    return t, recall
