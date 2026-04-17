from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


def session_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def load_manifest_filenames(manifest: Path) -> list[str]:
    rows: list[str] = []
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row["filename"])
    return rows


def build_session_groups(filenames: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for fn in filenames:
        groups[session_id_from_filename(fn)].append(fn)
    for sid in groups:
        groups[sid] = sorted(groups[sid])
    return dict(groups)


def session_train_val_files(
    filenames: list[str],
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    groups = build_session_groups(filenames)
    keys = sorted(groups.keys())
    train_s, val_s = train_test_split(keys, test_size=val_size, random_state=random_state, shuffle=True)
    train_files = [fn for sid in train_s for fn in groups[sid]]
    val_files = [fn for sid in val_s for fn in groups[sid]]
    return train_files, val_files
