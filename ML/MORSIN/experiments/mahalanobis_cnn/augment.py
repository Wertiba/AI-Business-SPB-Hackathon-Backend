"""Noise-resistant augmentations on log-mel tensors [3, F, T]."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def spec_freq_mask(mel: Tensor, max_width: int = 12) -> Tensor:
    """Randomly mask contiguous mel bands (per channel)."""
    c, f, t = mel.shape
    out = mel.clone()
    for i in range(c):
        if f <= 1 or torch.rand(1).item() > 0.5:
            continue
        w = int(torch.randint(1, min(max_width, f) + 1, (1,)).item())
        f0 = int(torch.randint(0, max(1, f - w), (1,)).item())
        out[i, f0 : f0 + w, :] = 0.0
    return out


def spec_time_mask(mel: Tensor, max_width: int = 24) -> Tensor:
    """Randomly mask contiguous time steps (all channels)."""
    c, f, t = mel.shape
    if t <= 1 or torch.rand(1).item() > 0.5:
        return mel
    w = int(torch.randint(1, min(max_width, t) + 1, (1,)).item())
    t0 = int(torch.randint(0, max(1, t - w), (1,)).item())
    out = mel.clone()
    out[:, :, t0 : t0 + w] = 0.0
    return out


def gaussian_noise(mel: Tensor, std: float = 0.03) -> Tensor:
    return mel + torch.randn_like(mel) * std


def augment_train(mel: Tensor) -> Tensor:
    x = mel
    x = gaussian_noise(x, std=0.02)
    x = spec_freq_mask(x)
    x = spec_time_mask(x)
    return x


def corrupt_for_synthetic_anomaly(mel: Tensor) -> Tensor:
    """Strong corruptions to mimic easy pseudo-anomalies for threshold tuning."""
    x = mel.clone()
    x = gaussian_noise(x, std=0.12)
    x = spec_freq_mask(x, max_width=28)
    x = spec_time_mask(x, max_width=48)
    # random gain per channel
    g = (0.4 + 0.8 * torch.rand(3, 1, 1, device=x.device, dtype=x.dtype))
    x = (x * g).clamp(-6.0, 6.0)
    return x
