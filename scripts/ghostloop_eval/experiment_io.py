"""Reproducible experiment controls shared by GhostLoop headless tools."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


CONSTRAINT_HEADER = [
    "enabled", "source_id", "target_id", "tx", "ty", "tz",
    "qx", "qy", "qz", "qw", "sigma_tx", "sigma_ty", "sigma_tz",
    "sigma_roll_deg", "sigma_pitch_deg", "sigma_yaw_deg",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_constraints_csv(path: Path, pose_count: int) -> list[list[str]]:
    """Load a frozen initial-loop checkpoint without changing numeric text."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty constraints CSV: {path}") from exc
        if header != CONSTRAINT_HEADER:
            raise ValueError(
                f"Unexpected constraints header in {path}: {header!r}"
            )
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No initial-loop constraints in {path}")
    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(CONSTRAINT_HEADER):
            raise ValueError(
                f"{path}:{line_number}: expected {len(CONSTRAINT_HEADER)} fields, "
                f"got {len(row)}"
            )
        if row[0] != "1":
            raise ValueError(
                f"{path}:{line_number}: frozen initial-loop row must be enabled"
            )
        source_id, target_id = int(row[1]), int(row[2])
        if not (0 <= source_id < pose_count and 0 <= target_id < pose_count):
            raise ValueError(
                f"{path}:{line_number}: pose ids {source_id}, {target_id} outside "
                f"[0, {pose_count})"
            )
        for value in row[3:]:
            float(value)
    return rows


def experiment_controls_payload(
    root_voxel_size: float,
    initial_constraints_csv: Path | None,
    initial_constraint_count: int,
) -> dict:
    return {
        "balm_root_voxel_size": float(root_voxel_size),
        "stage0_source": "fixed_csv" if initial_constraints_csv else "retrieval",
        "initial_constraints_csv": (
            str(initial_constraints_csv.resolve()) if initial_constraints_csv else None
        ),
        "initial_constraints_sha256": (
            sha256_file(initial_constraints_csv) if initial_constraints_csv else None
        ),
        "initial_constraint_count": int(initial_constraint_count),
    }


def load_balm_root_voxel_size(
    report_path: Path | None,
    explicit_value: float | None = None,
) -> float:
    if explicit_value is not None:
        value = float(explicit_value)
    elif report_path is not None:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        params = payload.get("params", payload.get("balm", {}))
        value = float(params["root_voxel_size"])
    else:
        raise ValueError("A BALM report or explicit root voxel size is required")
    if value <= 0.0:
        raise ValueError(f"root voxel size must be positive, got {value}")
    return value


def voxel_mask(
    points_xyz,
    voxel_keys: Iterable[Sequence[int]],
    root_voxel_size: float,
):
    """Return point membership for arbitrary-sized root voxels."""
    import numpy as np

    keys = np.floor(points_xyz / float(root_voxel_size)).astype(np.int64)
    keep = np.zeros(len(points_xyz), dtype=bool)
    for key in voxel_keys:
        keep |= np.all(keys == np.asarray(key, dtype=np.int64), axis=1)
    return keep


def validate_round_inputs(
    tums: Sequence[Path], labels: Sequence[str], loop_counts: Sequence[int]
) -> None:
    if not tums:
        raise ValueError("At least one round trajectory is required")
    if not (len(tums) == len(labels) == len(loop_counts)):
        raise ValueError(
            "--tums, --round-labels and --loop-counts must have equal lengths"
        )
