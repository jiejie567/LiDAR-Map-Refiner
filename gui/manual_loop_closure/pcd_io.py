from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from merge_pcds import read_pcd


class PcdValidationError(RuntimeError):
    """Raised when keyframe PCD files are missing or malformed."""


def list_numbered_pcds(directory: Path) -> List[Path]:
    numbered: List[tuple[int, Path]] = []
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".pcd":
            continue
        try:
            index = int(entry.stem)
        except ValueError:
            continue
        numbered.append((index, entry))

    if not numbered:
        raise PcdValidationError(f"No numbered PCD files found under {directory}")

    numbered.sort(key=lambda item: item[0])
    return [path for _, path in numbered]


def validate_keyframe_numbering(paths: List[Path], expected_count: int) -> None:
    indices = [int(path.stem) for path in paths]
    expected_indices = list(range(expected_count))
    if indices != expected_indices:
        raise PcdValidationError(
            "v1 requires keyframe files numbered exactly as 0.pcd..N-1.pcd. "
            f"Found {len(indices)} files with range [{indices[0]}..{indices[-1]}]."
        )


def load_xyz_points(path: Path) -> np.ndarray:
    cloud = read_pcd(path)
    required_fields = {"x", "y", "z"}
    if not required_fields.issubset(cloud.fields):
        raise PcdValidationError(
            f"PCD file is missing x/y/z fields: {path} fields={cloud.fields}"
        )

    points = np.column_stack(
        [
            cloud.data["x"].astype(np.float64, copy=False),
            cloud.data["y"].astype(np.float64, copy=False),
            cloud.data["z"].astype(np.float64, copy=False),
        ]
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise PcdValidationError(f"Unexpected point shape from {path}: {points.shape}")
    return points
