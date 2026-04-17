"""Prepare submission-time artifacts from the local training set and experiment checkpoint."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroice.deep import DeepMahalanobisScorer  # noqa: E402
from neuroice.features import FEATURE_VERSION, extract_features_from_audio  # noqa: E402
from neuroice.paths import CALIBRATION_PATH, CLASSICAL_ARTIFACT_PATH, DEEP_ARTIFACT_PATH  # noqa: E402


SOURCE_DEEP_ARTIFACT_PATH = ROOT / "experiments" / "artifacts" / "mahalanobis_resnet18.pt"
TRAIN_DIR = ROOT / "data" / "train"
MANIFEST_PATH = ROOT / "data" / "train.csv"
DEEP_WEIGHT = 0.7


def load_manifest_filenames(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["filename"] for row in reader]


def main() -> None:
    if not CLASSICAL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"missing classical artifact: {CLASSICAL_ARTIFACT_PATH}")
    if not SOURCE_DEEP_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"missing deep artifact: {SOURCE_DEEP_ARTIFACT_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")

    DEEP_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DEEP_ARTIFACT_PATH, DEEP_ARTIFACT_PATH)
    print(f"copied {SOURCE_DEEP_ARTIFACT_PATH.name} -> {DEEP_ARTIFACT_PATH.name}")

    bundle = joblib.load(CLASSICAL_ARTIFACT_PATH)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"Feature version mismatch: artifact={bundle.get('feature_version')!r} runtime={FEATURE_VERSION!r}"
        )

    scaler = bundle["scaler"]
    clf = bundle["clf"]
    deep = DeepMahalanobisScorer(DEEP_ARTIFACT_PATH)

    filenames = sorted(load_manifest_filenames(MANIFEST_PATH))
    classical_scores: list[float] = []
    deep_scores: list[float] = []

    for idx, filename in enumerate(filenames, start=1):
        path = TRAIN_DIR / filename
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        x = extract_features_from_audio(audio, int(sr)).reshape(1, -1)
        x = scaler.transform(x)
        classical_scores.append(float(-clf.decision_function(x)[0]))
        deep_scores.append(float(deep.score_path(path)))
        if idx % 50 == 0 or idx == len(filenames):
            print(f"scored {idx}/{len(filenames)}")

    payload = {
        "feature_version": FEATURE_VERSION,
        "deep_weight": DEEP_WEIGHT,
        "classical_sorted_scores": np.sort(np.asarray(classical_scores, dtype=np.float32)),
        "deep_sorted_scores": np.sort(np.asarray(deep_scores, dtype=np.float32)),
    }
    joblib.dump(payload, CALIBRATION_PATH)
    print(f"wrote {CALIBRATION_PATH} ({CALIBRATION_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
