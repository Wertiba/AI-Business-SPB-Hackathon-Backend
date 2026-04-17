"""Multi-resolution log-mel spectrograms from waveform (torch)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

_FBANK_CACHE: dict[tuple[int, int, int, str, torch.dtype], Tensor] = {}
_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def _cached_fbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    key = (n_fft, n_mels, sample_rate, str(device), dtype)
    if key not in _FBANK_CACHE:
        _FBANK_CACHE[key] = melscale_fbanks(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=50.0,
            f_max=float(sample_rate) / 2.0 * 0.99,
            device=device,
            dtype=dtype,
        )
    return _FBANK_CACHE[key]


def _cached_window(n_fft: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (n_fft, str(device), dtype)
    if key not in _WINDOW_CACHE:
        _WINDOW_CACHE[key] = torch.hann_window(n_fft, device=device, dtype=dtype)
    return _WINDOW_CACHE[key]


def _hz_to_mel(f: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: Tensor) -> Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def melscale_fbanks(
    *,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Shape (n_mels, n_fft // 2 + 1) triangular mel filterbank."""
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(torch.tensor(f_min, device=device, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(f_max, device=device, dtype=dtype))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz_pts = _mel_to_hz(mel_pts)
    bins = (hz_pts * (n_fft) / sample_rate).floor().long()
    bins = torch.clamp(bins, 0, n_freqs - 1)
    fb = torch.zeros(n_mels, n_freqs, device=device, dtype=dtype)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_freqs - 1)
        for f in range(left, center):
            fb[i, f] = (f - left) / max(center - left, 1)
        for f in range(center, right):
            fb[i, f] = (right - f) / max(right - center, 1)
    enorm = fb.sum(dim=1, keepdim=True).clamp(min=1e-6)
    fb = fb / enorm
    return fb


def _stft_mag(wav: Tensor, n_fft: int, hop: int, window: Tensor) -> Tensor:
    # wav [T]
    return torch.stft(
        wav,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()


def wav_to_logmel(
    wav: Tensor,
    sr: int,
    *,
    n_fft: int,
    hop: int,
    n_mels: int,
    fbank: Tensor,
    window: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """wav [T] float32 -> log-mel [n_mels, T_frames]."""
    mag = _stft_mag(wav, n_fft, hop, window).clamp_min(eps)
    mel = torch.matmul(fbank, mag)
    return torch.log(mel + eps)


def multi_resolution_logmel(
    wav_mono: Tensor,
    sr: int,
    *,
    n_mels: int,
    resolutions: Sequence[tuple[int, int]],
    target_time: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    wav_mono [T] on device.
    Returns tensor [3, n_mels, target_time] — one mel per channel at different (n_fft, hop).
    """
    channels: list[Tensor] = []
    for n_fft, hop in resolutions:
        fbank = _cached_fbank(n_fft, n_mels, sr, device, dtype)
        window = _cached_window(n_fft, device, dtype)
        lm = wav_to_logmel(wav_mono, sr, n_fft=n_fft, hop=hop, n_mels=n_mels, fbank=fbank, window=window)
        lm = (lm - lm.mean(dim=1, keepdim=True)) / (lm.std(dim=1, keepdim=True) + 1e-6)
        lm = F.interpolate(lm.unsqueeze(0).unsqueeze(0), size=(n_mels, target_time), mode="bilinear", align_corners=False)
        channels.append(lm.squeeze(0).squeeze(0))
    return torch.stack(channels, dim=0)


def load_wav_mono(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = torch.from_numpy(audio).to(device=device, dtype=dtype).mean(dim=1)
    return x, int(sr)


def mel_subwindows(mel: Tensor, n_sub: int = 4, min_width: int = 32) -> list[Tensor]:
    """mel [3, F, T] -> list of crops along time for subwindow encoding."""
    _, _, t = mel.shape
    if t <= min_width:
        return [mel]
    width = max(min_width, int(math.ceil(t / (n_sub + 1))))
    if width >= t:
        return [mel]
    starts: list[int] = []
    step = max(1, (t - width) // max(n_sub - 1, 1))
    for i in range(n_sub):
        s = min(i * step, t - width)
        starts.append(int(s))
    seen: set[int] = set()
    out: list[Tensor] = []
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        out.append(mel[:, :, s : s + width])
    return out if out else [mel]


def clip_logmel_from_path(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    n_mels: int = 128,
    target_time: int = 256,
    resolutions: Sequence[tuple[int, int]] = ((1024, 256), (2048, 512), (4096, 1024)),
) -> Tensor:
    wav, sr = load_wav_mono(path, device, dtype)
    return multi_resolution_logmel(
        wav,
        sr,
        n_mels=n_mels,
        resolutions=resolutions,
        target_time=target_time,
        device=device,
        dtype=dtype,
    )
