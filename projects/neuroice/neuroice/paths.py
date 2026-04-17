from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

CLASSICAL_ARTIFACT_PATH = ARTIFACTS_DIR / "classical.joblib"
DEEP_ARTIFACT_PATH = ARTIFACTS_DIR / "deep.pt"
CALIBRATION_PATH = ARTIFACTS_DIR / "score_calibration.joblib"
