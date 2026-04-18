import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import joblib

def get_embedding(audio, sr):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                  n_fft=1024, hop_length=256)
    return np.concatenate([np.mean(mfcc, axis=1),
                            np.std(mfcc, axis=1)])

wav_files = sorted(Path("data/train").glob("*.wav"))
embeddings = []
for f in wav_files:
    audio, sr = sf.read(str(f), always_2d=True)
    embeddings.append(get_embedding(audio, sr))

X = np.array(embeddings)
knn = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=-1)
knn.fit(X)

joblib.dump(knn, "knn.pkl")
print(f"KNN обучен на {len(X)} файлах")
