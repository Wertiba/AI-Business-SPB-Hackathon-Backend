from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from neuroice.features import extract_features_from_audio
from solution import PredictionModel


ROOT = Path(__file__).resolve().parents[1]


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_shape_is_stable(self) -> None:
        audio = np.zeros((48_000, 2), dtype=np.float32)
        feats = extract_features_from_audio(audio, 48_000)
        self.assertEqual(feats.shape, (47,))
        self.assertTrue(np.isfinite(feats).all())


class PredictionModelSmokeTests(unittest.TestCase):
    def test_predict_returns_finite_scores_for_local_train_files(self) -> None:
        batch = sorted((ROOT / "data" / "train").glob("*.wav"))[:3]
        self.assertEqual(len(batch), 3, "expected local training WAVs for smoke test")

        model = PredictionModel()
        scores = model.predict(batch)

        self.assertEqual(len(scores), len(batch))
        self.assertTrue(all(isinstance(score, float) for score in scores))
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))


if __name__ == "__main__":
    unittest.main()
