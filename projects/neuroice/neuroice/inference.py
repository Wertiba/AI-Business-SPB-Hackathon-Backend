from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from .features import FEATURE_VERSION, extract_features_from_audio
from .paths import CLASSICAL_ARTIFACT_PATH


class EmpiricalTailCalibrator:
    def __init__(self, sorted_scores: np.ndarray) -> None:
        scores = np.asarray(sorted_scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            raise ValueError("sorted_scores must be non-empty")
        self._scores = np.sort(scores)
        self._denom = float(scores.size + 1)

    def transform(self, score: float) -> float:
        idx = np.searchsorted(self._scores, float(score), side="right")
        return float((idx + 0.5) / self._denom)


class PredictionModel:
    batch_size: int = 10

    def __init__(self) -> None:
        bundle = joblib.load(CLASSICAL_ARTIFACT_PATH)
        artifact_feature_version = bundle.get("feature_version")
        if artifact_feature_version != FEATURE_VERSION:
            raise RuntimeError(
                f"Feature version mismatch: artifact={artifact_feature_version!r} runtime={FEATURE_VERSION!r}"
            )

        self._scaler = bundle["scaler"]
        self._pca = bundle["pca"]
        self._pca_calibrator = EmpiricalTailCalibrator(bundle["pca_sorted_scores"])

    def _extract_scaled_features(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        x = extract_features_from_audio(audio, int(sr)).reshape(1, -1)
        return self._scaler.transform(x).astype(np.float64, copy=False)

    def _score_pca_raw(self, x_scaled: np.ndarray) -> float:
        z = self._pca.transform(x_scaled)
        recon = self._pca.inverse_transform(z)
        return float(np.sum((x_scaled - recon) ** 2, axis=1)[0])

    def _score_pca(self, path: Path) -> float:
        x_scaled = self._extract_scaled_features(path)
        return self._pca_calibrator.transform(self._score_pca_raw(x_scaled))

    def predict(self, batch: list[Path]) -> list[float]:
        scores: list[float] = []
        for path in batch:
            try:
                scores.append(float(self._score_pca(path)))
            except Exception:
                scores.append(1.0)
        return scores
