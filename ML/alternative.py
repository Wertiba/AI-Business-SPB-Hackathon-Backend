from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import joblib

def get_embedding(audio, sr):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                  n_fft=1024, hop_length=256)
    return np.concatenate([np.mean(mfcc, axis=1),
                            np.std(mfcc, axis=1)])

class PredictionModel:
    batch_size: int = 10

    def __init__(self) -> None:
        base = Path(__file__).parent
        self.knn = joblib.load(base / "knn.pkl")

    def predict(self, batch: list[Path]) -> list[float]:
        scores = []
        for path in batch:
            try:
                audio, sr = sf.read(str(path), always_2d=True)
                mono = audio.mean(axis=1).astype(np.float64)

                rms = float(np.sqrt(np.mean(mono ** 2)))

                emb = get_embedding(audio, sr).reshape(1, -1)
                dists, _ = self.knn.kneighbors(emb)
                knn_score = float(dists.mean())

                score = 0.5 * rms + 0.5 * knn_score
            except Exception:
                score = 0.0
            scores.append(score)
        return scores
