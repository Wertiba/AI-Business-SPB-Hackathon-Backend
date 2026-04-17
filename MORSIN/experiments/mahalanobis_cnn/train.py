"""Train ResNet18 on multi-resolution log-mel with contrastive loss,
fit Mahalanobis on embeddings (group-based val, synthetic anomalies for Recall@FPR5),
then refit Mahalanobis on full data and save artifacts under experiments/artifacts/.

Dependencies: ``torch`` + ``torchvision`` (optional extra: ``uv sync --extra experiments``,
or ``pip install torch torchvision`` into the project venv).

ImageNet weights download to ``<repo>/.torch_home`` (writable, gitignored). Use
``--no-pretrained`` when offline.

Run from repo root::

  .venv/bin/python -m experiments.mahalanobis_cnn.train [--epochs 12] [--embed-dim 128|256]
      [--no-pretrained]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mahalanobis_cnn.augment import augment_train, corrupt_for_synthetic_anomaly
from experiments.mahalanobis_cnn.encoder import MelResNetEncoder
from experiments.mahalanobis_cnn.groups import load_manifest_filenames, session_train_val_files
from experiments.mahalanobis_cnn.mahalanobis import MahalanobisScorer, recall_at_fpr_threshold
from experiments.mahalanobis_cnn.spec import clip_logmel_from_path, mel_subwindows


def embed_clip(encoder: MelResNetEncoder, mel: Tensor, *, n_sub: int = 4) -> Tensor:
    """mel [3,F,T] -> L2-normalized mean of subwindow embeddings [D]."""
    encoder.eval()
    crops = mel_subwindows(mel, n_sub=n_sub, min_width=32)
    zs = []
    with torch.no_grad():
        for c in crops:
            zs.append(encoder(c.unsqueeze(0)).squeeze(0))
    zstack = torch.stack(zs, dim=0)
    return F.normalize(zstack.mean(dim=0, keepdim=True), dim=1).squeeze(0)


def contrastive_clip_loss(z1: Tensor, z2: Tensor, *, temperature: float = 0.15) -> Tensor:
    """Symmetric InfoNCE with in-batch negatives (CLIP-style)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = (z1 @ z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class MelPathDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Path:
        return self.paths[idx]


def collate_paths(batch: list[Path]) -> list[Path]:
    return batch


def collate_contrastive(
    batch_paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    v1_list: list[Tensor] = []
    v2_list: list[Tensor] = []
    for p in batch_paths:
        mel = clip_logmel_from_path(p, device, dtype)
        v1_list.append(augment_train(mel))
        v2_list.append(augment_train(mel))
    return torch.stack(v1_list, dim=0), torch.stack(v2_list, dim=0)


@torch.no_grad()
def encode_paths(
    encoder: MelResNetEncoder,
    paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_sub: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in paths:
        mel = clip_logmel_from_path(p, device, dtype)
        z = embed_clip(encoder, mel, n_sub=n_sub)
        out.append(z.detach().cpu().numpy().astype(np.float32))
    return np.stack(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 mel + Mahalanobis experiment")
    parser.add_argument("--epochs", type=int, default=12, help="Contrastive training epochs")
    parser.add_argument("--embed-dim", type=int, default=128, choices=(128, 256))
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random init (no ImageNet weights); use offline / no-cache setups",
    )
    args = parser.parse_args()

    torch_home = ROOT / ".torch_home"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    data_dir = ROOT / "data"
    train_dir = data_dir / "train"
    manifest = data_dir / "train.csv"
    if not manifest.exists():
        print(f"ERROR: missing {manifest}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    embed_dim = args.embed_dim
    n_sub = 4
    epochs = args.epochs
    batch_size = 16 if device.type == "cuda" else 6
    lr = 3e-4

    filenames = sorted(load_manifest_filenames(manifest))
    train_files, val_files = session_train_val_files(filenames, val_size=0.2, random_state=42)
    train_paths = [train_dir / f for f in train_files]
    val_paths = [train_dir / f for f in val_files]

    print(f"device={device} train_chunks={len(train_paths)} val_chunks={len(val_paths)}")

    encoder = MelResNetEncoder(embed_dim=embed_dim, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs), eta_min=1e-6
    )

    ds = MelPathDataset(train_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_paths,
    )

    encoder.train()
    for ep in range(epochs):
        lr_epoch = float(opt.param_groups[0]["lr"])
        losses: list[float] = []
        for batch_paths in loader:
            m1, m2 = collate_contrastive(batch_paths, device=device, dtype=dtype)
            z1 = encoder(m1)
            z2 = encoder(m2)
            loss = contrastive_clip_loss(z1, z2, temperature=0.15)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        print(
            f"epoch {ep + 1}/{epochs} loss_mean={float(np.mean(losses)):.4f} "
            f"lr_used={lr_epoch:.2e}"
        )

    # --- embeddings + Mahalanobis on train sessions; validate on held-out sessions ---
    encoder.eval()
    X_tr = encode_paths(encoder, train_paths, device=device, dtype=dtype, n_sub=n_sub)
    X_va = encode_paths(encoder, val_paths, device=device, dtype=dtype, n_sub=n_sub)

    scorer_diag = MahalanobisScorer().fit(X_tr)
    s_tr = scorer_diag.score_samples(X_tr)
    s_va = scorer_diag.score_samples(X_va)

    print("--- Mahalanobis (fit on train-session embeddings only) ---")
    print(
        f"  train_normal: mean={s_tr.mean():.4f} std={s_tr.std():.4f} "
        f"min={s_tr.min():.4f} max={s_tr.max():.4f}"
    )
    print(
        f"  val_normal  : mean={s_va.mean():.4f} std={s_va.std():.4f} "
        f"min={s_va.min():.4f} max={s_va.max():.4f}"
    )

    # Synthetic pseudo-anomalies from val mel (corrupted); threshold tuned on val normals @ FPR=5%
    synth_scores: list[float] = []
    for p in val_paths:
        mel = clip_logmel_from_path(p, device, dtype)
        mel_c = corrupt_for_synthetic_anomaly(mel)
        z = embed_clip(encoder, mel_c, n_sub=n_sub)
        # Mahalanobis in train Gaussian; use delta from train fit
        s = float(scorer_diag.score_samples(z.detach().cpu().numpy().reshape(1, -1))[0])
        synth_scores.append(s)
    synth_scores_np = np.asarray(synth_scores, dtype=np.float64)

    t_fpr5, rec_fpr5 = recall_at_fpr_threshold(s_va, synth_scores_np, fpr=0.05)
    print(f"--- Recall@FPR5 (synthetic val anomalies; threshold from val normals) ---")
    print(f"  threshold_t@FPR5={t_fpr5:.6f}  recall_on_synthetic={rec_fpr5:.4f}")

    # --- final: all chunks, refit Mahalanobis ---
    all_paths = [train_dir / f for f in filenames]
    X_all = encode_paths(encoder, all_paths, device=device, dtype=dtype, n_sub=n_sub)
    scorer_final = MahalanobisScorer().fit(X_all)
    s_all = scorer_final.score_samples(X_all)
    print(f"--- final fit on ALL chunks (n={len(all_paths)}) ---")
    print(f"  mahalanobis_train_normal mean={s_all.mean():.4f} std={s_all.std():.4f}")

    out_dir = ROOT / "experiments" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "encoder_state": encoder.state_dict(),
        "embed_dim": embed_dim,
        "n_sub": n_sub,
        "mahalanobis_location": scorer_final.location_,
        "mahalanobis_precision": scorer_final.precision_,
        "n_mels": 128,
        "target_time": 256,
        "resolutions": [(1024, 256), (2048, 512), (4096, 1024)],
        "val_fpr5_threshold_diag": t_fpr5,
        "val_recall_synthetic_diag": rec_fpr5,
    }
    torch.save(ckpt, out_dir / "mahalanobis_resnet18.pt")
    meta = {
        "embed_dim": embed_dim,
        "n_sub": n_sub,
        "epochs": epochs,
        "backbone": "resnet18",
        "pretrained": not args.no_pretrained,
        "device": str(device),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "val_fpr5_threshold_diag": t_fpr5,
        "val_recall_synthetic_at_fpr5": rec_fpr5,
    }
    (out_dir / "mahalanobis_resnet18_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'mahalanobis_resnet18.pt'}")
    print(f"wrote {out_dir / 'mahalanobis_resnet18_meta.json'}")


if __name__ == "__main__":
    main()
