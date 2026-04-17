"""Shared audio feature extraction for training and inference."""

from __future__ import annotations

import numpy as np


FEATURE_VERSION = "v1"


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a 1D float32 feature vector for one clip."""
    x_all = np.asarray(audio, dtype=np.float32)
    if x_all.ndim == 1:
        x_all = x_all[:, np.newaxis]

    feats: list[float] = []

    for c in range(x_all.shape[1]):
        x = x_all[:, c]
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        zcr = float(np.mean((x[:-1] * x[1:]) < 0)) if x.size > 1 else 0.0
        feats.extend([rms, peak, peak / rms, zcr])

    mono = np.mean(x_all, axis=1)
    rms_m = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak_m = float(np.max(np.abs(mono)) + 1e-12)
    zcr_m = float(np.mean((mono[:-1] * mono[1:]) < 0)) if mono.size > 1 else 0.0
    feats.extend([rms_m, peak_m, peak_m / rms_m, zcr_m])

    n_fft = 2048
    hop = 512
    sig = mono
    if sig.size < n_fft:
        sig = np.pad(sig, (0, n_fft - sig.size))

    n_frames = 1 + (sig.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    mags: list[np.ndarray] = []
    for i in range(n_frames):
        frame = sig[i * hop : i * hop + n_fft] * window
        mags.append(np.abs(np.fft.rfft(frame)))
    mag = np.stack(mags, axis=1)

    log_mag = np.log(mag + 1e-10)
    mean_log = np.mean(log_mag, axis=1)
    n_bins = mean_log.size
    freqs = np.linspace(0.0, float(sr) / 2.0, n_bins, dtype=np.float32)

    mel_lo = _hz_to_mel(np.array([50.0], dtype=np.float64))[0]
    mel_hi = _hz_to_mel(np.array([float(sr) / 2.0 * 0.99], dtype=np.float64))[0]
    mel_edges = np.linspace(mel_lo, mel_hi, 33, dtype=np.float64)
    hz_edges = _mel_to_hz(mel_edges).astype(np.float32)

    n_mel_bands = 32
    for i in range(n_mel_bands):
        lo, hi = hz_edges[i], hz_edges[i + 1]
        if i < n_mel_bands - 1:
            mask = (freqs >= lo) & (freqs < hi)
        else:
            mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            feats.append(0.0)
        else:
            feats.append(float(np.mean(mean_log[mask])))

    pow_spec = np.exp(2.0 * mean_log)
    w = pow_spec / (np.sum(pow_spec) + 1e-12)
    centroid = float(np.sum(freqs.astype(np.float64) * w))
    spread = float(np.sqrt(np.sum(((freqs.astype(np.float64) - centroid) ** 2) * w)))
    flatness = float(np.exp(np.mean(mean_log)) / (np.mean(np.exp(mean_log)) + 1e-12))
    feats.extend([centroid, spread, flatness])

    return np.asarray(feats, dtype=np.float32)
