from pathlib import Path

import numpy as np
import soundfile as sf


class PredictionModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def predict(batch: list[Path]) -> list[float]:
        scores = []
        for path in batch:
            try:
                audio, _ = sf.read(str(path), always_2d=True)
                mono = audio.mean(axis=1).astype(np.float64)

                score = float(np.percentile(np.abs(mono), 99))
            except Exception:   # noqa
                score = 0.0
            scores.append(score)
        return scores
