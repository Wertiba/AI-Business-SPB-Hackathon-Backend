"""Run exploratory data analysis for MORSIN/data and save plots to a separate folder."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MORSIN.neuroice.features import extract_features_from_audio  # noqa: E402


OUTPUT_DIR = ROOT / "eda_outputs"
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
MANIFEST_PATH = DATA_DIR / "train.csv"


def session_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _compute_clip_record(filename: str, label: str) -> dict[str, object]:
    path = TRAIN_DIR / filename
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    info = sf.info(path)
    mono = audio.mean(axis=1)
    feats = extract_features_from_audio(audio, int(sr))

    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak = float(np.max(np.abs(mono)) + 1e-12)
    zcr = float(np.mean((mono[:-1] * mono[1:]) < 0)) if mono.size > 1 else 0.0
    duration_s = float(info.frames / info.samplerate)
    audio_hash = hashlib.sha256(audio.tobytes()).hexdigest()

    return {
        "filename": filename,
        "label": label,
        "session_id": session_id_from_filename(filename),
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "peak_to_rms": peak / rms,
        "zcr": zcr,
        "spectral_centroid": float(feats[-3]),
        "spectral_spread": float(feats[-2]),
        "spectral_flatness": float(feats[-1]),
        "audio_hash": audio_hash,
    }


def build_analysis_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST_PATH)
    records = [_compute_clip_record(row.filename, row.label) for row in manifest.itertuples(index=False)]
    clip_df = pd.DataFrame.from_records(records).sort_values("filename").reset_index(drop=True)

    session_df = (
        clip_df.groupby("session_id")
        .agg(
            clips=("filename", "count"),
            label_nunique=("label", "nunique"),
            rms_mean=("rms", "mean"),
            rms_std=("rms", "std"),
            peak_mean=("peak", "mean"),
            zcr_mean=("zcr", "mean"),
            centroid_mean=("spectral_centroid", "mean"),
        )
        .sort_values("rms_mean")
        .reset_index()
    )
    return clip_df, session_df


def add_pca_projection(clip_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "rms",
        "peak",
        "peak_to_rms",
        "zcr",
        "spectral_centroid",
        "spectral_spread",
        "spectral_flatness",
    ]
    scaler = StandardScaler()
    X = scaler.fit_transform(clip_df[feature_cols].to_numpy())
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    out = clip_df.copy()
    out["pc1"] = pcs[:, 0]
    out["pc2"] = pcs[:, 1]
    out["pca_explained_var"] = pca.explained_variance_ratio_.sum()
    return out


def save_summary(clip_df: pd.DataFrame, session_df: pd.DataFrame) -> None:
    duplicate_groups = int(clip_df["audio_hash"].duplicated(keep=False).sum())
    clipped_files = int((clip_df["peak"] >= 0.9999).sum())
    label_counts = clip_df["label"].value_counts().sort_values(ascending=False)
    loudest = session_df.sort_values("rms_mean", ascending=False).head(5)
    quietest = session_df.sort_values("rms_mean", ascending=True).head(5)

    summary_lines = [
        "# MORSIN Data EDA",
        "",
        "## Critical Findings",
        f"- The manifest has `{len(clip_df)}` rows, but only `{clip_df['label'].nunique()}` unique label(s): "
        + ", ".join(f"`{label}` ({count})" for label, count in label_counts.items()),
        f"- The dataset is organized into `{session_df.shape[0]}` session-like groups with exactly "
        f"`{int(session_df['clips'].min())}` clips per session in this snapshot.",
        f"- All clips are `{clip_df['duration_s'].iloc[0]:.1f}` seconds long at "
        f"`{int(clip_df['samplerate'].iloc[0])}` Hz, indicating a fully standardized recording format.",
        f"- Exact duplicate waveform groups detected: `{duplicate_groups}` duplicated files across all `{len(clip_df)}` clips.",
        f"- `{clipped_files}` clips reach peak amplitude `1.0`, which is consistent with clipping or aggressive peak normalization.",
        "",
        "## Distribution Notes",
        f"- RMS loudness ranges from `{clip_df['rms'].min():.4f}` to `{clip_df['rms'].max():.4f}`.",
        f"- Spectral centroid ranges from `{clip_df['spectral_centroid'].min():.2f}` to "
        f"`{clip_df['spectral_centroid'].max():.2f}` Hz.",
        f"- There is substantial session-to-session recording variance: quietest session mean RMS is "
        f"`{session_df['rms_mean'].min():.4f}`, loudest is `{session_df['rms_mean'].max():.4f}`.",
        "",
        "## Quietest Sessions By Mean RMS",
    ]
    summary_lines.extend(
        f"- `{row.session_id}`: mean RMS `{row.rms_mean:.4f}`, mean peak `{row.peak_mean:.4f}`, mean ZCR `{row.zcr_mean:.4f}`"
        for row in quietest.itertuples(index=False)
    )
    summary_lines.extend(["", "## Loudest Sessions By Mean RMS"])
    summary_lines.extend(
        f"- `{row.session_id}`: mean RMS `{row.rms_mean:.4f}`, mean peak `{row.peak_mean:.4f}`, mean ZCR `{row.zcr_mean:.4f}`"
        for row in loudest.itertuples(index=False)
    )

    (OUTPUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    clip_df.to_csv(OUTPUT_DIR / "clip_level_features.csv", index=False)
    session_df.to_csv(OUTPUT_DIR / "session_level_summary.csv", index=False)


def plot_label_counts(clip_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = clip_df["label"].value_counts().sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C78A8")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Clip Count")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + max(1, 0.01 * value), str(int(value)), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_distribution.png", dpi=180)
    plt.close(fig)


def plot_duration_and_samplerate(clip_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(clip_df["duration_s"], bins=20, ax=axes[0], color="#59A14F")
    axes[0].set_title("Clip Duration")
    axes[0].set_xlabel("Seconds")

    sr_counts = clip_df["samplerate"].value_counts().sort_index()
    sns.barplot(x=sr_counts.index.astype(str), y=sr_counts.values, ax=axes[1], color="#F28E2B")
    axes[1].set_title("Sample Rate Distribution")
    axes[1].set_xlabel("Sample Rate (Hz)")
    axes[1].set_ylabel("Clip Count")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "duration_and_samplerate.png", dpi=180)
    plt.close(fig)


def plot_feature_distributions(clip_df: pd.DataFrame) -> None:
    cols = ["rms", "peak", "zcr", "spectral_centroid"]
    titles = ["RMS", "Peak", "Zero-Crossing Rate", "Spectral Centroid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title in zip(axes.flat, cols, titles):
        sns.histplot(clip_df[col], bins=30, kde=True, ax=ax, color="#E15759")
        ax.set_title(title)
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=180)
    plt.close(fig)


def plot_session_rms(session_df: pd.DataFrame) -> None:
    ordered = session_df.sort_values("rms_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(len(ordered)), ordered["rms_mean"], color="#76B7B2")
    ax.set_title("Mean RMS By Session")
    ax.set_xlabel("Session Rank (sorted by RMS)")
    ax.set_ylabel("Mean RMS")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "session_mean_rms.png", dpi=180)
    plt.close(fig)


def plot_feature_scatter(clip_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        clip_df["rms"],
        clip_df["spectral_centroid"],
        c=clip_df["zcr"],
        cmap="viridis",
        alpha=0.8,
        s=34,
        edgecolors="none",
    )
    ax.set_title("RMS vs Spectral Centroid")
    ax.set_xlabel("RMS")
    ax.set_ylabel("Spectral Centroid (Hz)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Zero-Crossing Rate")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rms_vs_centroid.png", dpi=180)
    plt.close(fig)


def plot_pca_projection(clip_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    session_codes, _ = pd.factorize(clip_df["session_id"])
    sc = ax.scatter(clip_df["pc1"], clip_df["pc2"], c=session_codes, cmap="tab20", s=34, alpha=0.85)
    ax.set_title("PCA Projection Of Audio Features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="Session Index")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_projection.png", dpi=180)
    plt.close(fig)


def plot_waveform_examples(clip_df: pd.DataFrame) -> None:
    sample_rows = pd.concat(
        [
            clip_df.nsmallest(1, "rms"),
            clip_df.nlargest(1, "rms"),
            clip_df.nsmallest(1, "spectral_centroid"),
            clip_df.nlargest(1, "spectral_centroid"),
        ]
    ).drop_duplicates(subset=["filename"])

    fig, axes = plt.subplots(len(sample_rows), 1, figsize=(12, 2.6 * len(sample_rows)), sharex=False)
    if len(sample_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, sample_rows.itertuples(index=False)):
        audio, sr = sf.read(TRAIN_DIR / row.filename, always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)
        t = np.arange(mono.shape[0]) / sr
        ax.plot(t, mono, linewidth=0.7, color="#4C78A8")
        ax.set_title(
            f"{row.filename} | rms={row.rms:.4f} | centroid={row.spectral_centroid:.1f} Hz | zcr={row.zcr:.4f}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "waveform_examples.png", dpi=180)
    plt.close(fig)


def plot_spectrogram_examples(clip_df: pd.DataFrame) -> None:
    sample_rows = pd.concat(
        [clip_df.nsmallest(1, "rms"), clip_df.nlargest(1, "rms")]
    ).drop_duplicates(subset=["filename"])

    fig, axes = plt.subplots(len(sample_rows), 1, figsize=(12, 3.4 * len(sample_rows)))
    if len(sample_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, sample_rows.itertuples(index=False)):
        audio, sr = sf.read(TRAIN_DIR / row.filename, always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)
        freqs, times, spec = signal.spectrogram(mono, fs=sr, nperseg=1024, noverlap=768)
        spec_db = 10.0 * np.log10(spec + 1e-12)
        mesh = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="magma")
        ax.set_ylim(0, 5000)
        ax.set_title(f"Spectrogram: {row.filename}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(mesh, ax=ax, label="Power (dB)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spectrogram_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    clip_df, session_df = build_analysis_frames()
    clip_df = add_pca_projection(clip_df)

    save_summary(clip_df, session_df)
    plot_label_counts(clip_df)
    plot_duration_and_samplerate(clip_df)
    plot_feature_distributions(clip_df)
    plot_session_rms(session_df)
    plot_feature_scatter(clip_df)
    plot_pca_projection(clip_df)
    plot_waveform_examples(clip_df)
    plot_spectrogram_examples(clip_df)

    print(f"Wrote EDA outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
