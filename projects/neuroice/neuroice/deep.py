"""Deep log-mel + ResNet18 + Mahalanobis inference utilities for submission."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
import torch.nn as nn
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
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(torch.tensor(f_min, device=device, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(f_max, device=device, dtype=dtype))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz_pts = _mel_to_hz(mel_pts)
    bins = (hz_pts * n_fft / sample_rate).floor().long()
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
    return fb / fb.sum(dim=1, keepdim=True).clamp(min=1e-6)


def _stft_mag(wav: Tensor, n_fft: int, hop: int, window: Tensor) -> Tensor:
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
    channels: list[Tensor] = []
    for n_fft, hop in resolutions:
        fbank = _cached_fbank(n_fft, n_mels, sr, device, dtype)
        window = _cached_window(n_fft, device, dtype)
        lm = wav_to_logmel(wav_mono, sr, n_fft=n_fft, hop=hop, n_mels=n_mels, fbank=fbank, window=window)
        lm = (lm - lm.mean(dim=1, keepdim=True)) / (lm.std(dim=1, keepdim=True) + 1e-6)
        lm = F.interpolate(
            lm.unsqueeze(0).unsqueeze(0),
            size=(n_mels, target_time),
            mode="bilinear",
            align_corners=False,
        )
        channels.append(lm.squeeze(0).squeeze(0))
    return torch.stack(channels, dim=0)


def clip_logmel_from_path(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    n_mels: int,
    target_time: int,
    resolutions: Sequence[tuple[int, int]],
) -> Tensor:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(audio).to(device=device, dtype=dtype).mean(dim=1)
    return multi_resolution_logmel(
        wav,
        int(sr),
        n_mels=n_mels,
        resolutions=resolutions,
        target_time=target_time,
        device=device,
        dtype=dtype,
    )


def mel_subwindows(mel: Tensor, n_sub: int = 4, min_width: int = 32) -> list[Tensor]:
    _, _, t = mel.shape
    if t <= min_width:
        return [mel]
    width = max(min_width, int(math.ceil(t / (n_sub + 1))))
    if width >= t:
        return [mel]
    step = max(1, (t - width) // max(n_sub - 1, 1))
    starts = [min(i * step, t - width) for i in range(n_sub)]
    out: list[Tensor] = []
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        out.append(mel[:, :, start : start + width])
    return out or [mel]


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class MelResNetEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        backbone = ResNet18Backbone()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = backbone.avgpool
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, mel: Tensor) -> Tensor:
        x = F.interpolate(mel, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.fc(x), dim=1)


class DeepMahalanobisScorer:
    def __init__(self, checkpoint_path: Path, *, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.n_mels = int(checkpoint["n_mels"])
        self.target_time = int(checkpoint["target_time"])
        self.resolutions = tuple(tuple(int(v) for v in pair) for pair in checkpoint["resolutions"])
        self.n_sub = int(checkpoint["n_sub"])

        self.encoder = MelResNetEncoder(embed_dim=int(checkpoint["embed_dim"])).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.encoder.eval()

        self.location = torch.as_tensor(checkpoint["mahalanobis_location"], device=self.device, dtype=self.dtype)
        self.precision = torch.as_tensor(checkpoint["mahalanobis_precision"], device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def embed_path(self, path: Path) -> Tensor:
        mel = clip_logmel_from_path(
            path,
            self.device,
            self.dtype,
            n_mels=self.n_mels,
            target_time=self.target_time,
            resolutions=self.resolutions,
        )
        crops = mel_subwindows(mel, n_sub=self.n_sub, min_width=32)
        zs = [self.encoder(c.unsqueeze(0)).squeeze(0) for c in crops]
        return F.normalize(torch.stack(zs, dim=0).mean(dim=0, keepdim=True), dim=1).squeeze(0)

    @torch.inference_mode()
    def score_path(self, path: Path) -> float:
        z = self.embed_path(path)
        delta = z - self.location
        quad = torch.sum((delta @ self.precision) * delta)
        return float(torch.sqrt(torch.clamp(quad, min=0.0)).detach().cpu())
