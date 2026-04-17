"""Fit a PCA reconstruction anomaly model on normal train WAVs and export bundle artifacts."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroice.features import FEATURE_VERSION, extract_features_from_audio  # noqa: E402
from neuroice.paths import CLASSICAL_ARTIFACT_PATH  # noqa: E402

PCA_COMPONENTS = 24


def session_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def load_manifest_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filename"], row.get("label", "normal")))
    return rows


def build_session_groups(rows: list[tuple[str, str]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for filename, _label in rows:
        sid = session_id_from_filename(filename)
        groups[sid].append(filename)
    return dict(groups)


def _score_summary(name: str, scores: np.ndarray) -> None:
    s = scores.astype(np.float64)
    print(
        f"  {name}: n={s.size} min={float(np.min(s)):.4f} max={float(np.max(s)):.4f} "
        f"mean={float(np.mean(s)):.4f} median={float(np.median(s)):.4f} std={float(np.std(s)):.4f}"
    )


def pca_reconstruction_scores(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    n_components: int,
) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=min(n_components, X_train.shape[1] - 1), random_state=42)
    pca.fit(X_train)
    recon = pca.inverse_transform(pca.transform(X_eval))
    scores = np.sum((X_eval - recon) ** 2, axis=1)
    return pca, scores


def main() -> None:
    data_dir = ROOT / "data"
    train_dir = data_dir / "train"
    manifest = data_dir / "train.csv"
    if not manifest.exists():
        print(f"ERROR: missing {manifest}", file=sys.stderr)
        sys.exit(1)

    rows = load_manifest_rows(manifest)
    groups = build_session_groups(rows)
    session_keys = sorted(groups.keys())
    train_sessions, val_sessions = train_test_split(
        session_keys,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    train_files = [fn for sid in train_sessions for fn in sorted(groups[sid])]
    val_files = [fn for sid in val_sessions for fn in sorted(groups[sid])]

    print(f"sessions total={len(session_keys)} train={len(train_sessions)} val={len(val_sessions)}")
    print(f"chunks train={len(train_files)} val={len(val_files)}")

    def feats_for_split(filenames: list[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for fn in filenames:
            path = train_dir / fn
            audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
            out.append(extract_features_from_audio(audio, int(sr)))
        return np.stack(out, axis=0)

    X_train_split = feats_for_split(train_files)
    X_val_split = feats_for_split(val_files)

    scaler_diag = StandardScaler().fit(X_train_split)
    X_tr_s = scaler_diag.transform(X_train_split)
    X_va_s = scaler_diag.transform(X_val_split)

    pca_diag, tr_pca = pca_reconstruction_scores(X_tr_s, X_tr_s, PCA_COMPONENTS)
    va_pca = np.sum((X_va_s - pca_diag.inverse_transform(pca_diag.transform(X_va_s))) ** 2, axis=1)

    print("--- validation (session holdout; PCA reconstruction fit on train sessions only) ---")
    _score_summary("train_pca   ", tr_pca)
    _score_summary("val_pca     ", va_pca)
    shift = float(np.mean(va_pca) - np.mean(tr_pca))
    print(f"  val_minus_train_mean: {shift:+.4f} (closer to 0 is better for i.i.d. normal chunks)")
    for q in (90, 95, 99):
        thr = float(np.percentile(tr_pca, q))
        frac = float(np.mean(va_pca > thr))
        print(f"  val_frac_above_train_p{q}: {frac:.4f}  (train p{q} threshold={thr:.4f})")

    all_files = sorted([fn for fn, _ in rows])
    X_all = feats_for_split(all_files)
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)
    pca, pca_scores = pca_reconstruction_scores(X_scaled, X_scaled, PCA_COMPONENTS)

    print(f"--- final fit on ALL normal chunks (n={len(all_files)}) -> exporting artifacts ---")

    artifact_path = CLASSICAL_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "feature_version": FEATURE_VERSION,
            "pca": pca,
            "pca_sorted_scores": np.sort(pca_scores.astype(np.float32)),
            "selected_model": "pca_reconstruction",
        },
        artifact_path,
    )
    print(f"wrote {artifact_path} ({artifact_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
