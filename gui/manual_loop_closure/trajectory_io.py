from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryValidationError(RuntimeError):
    """Raised when the optimized TUM trajectory is invalid."""


@dataclass(frozen=True)
class TrajectoryData:
    path: Path
    timestamps: np.ndarray
    positions_xyz: np.ndarray
    quats_xyzw: np.ndarray
    transforms_world_sensor: np.ndarray

    @property
    def size(self) -> int:
        return int(self.timestamps.shape[0])


def build_transform(
    translation_xyz: Sequence[float],
    quat_xyzw: Sequence[float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return transform


def matrix_to_quat_xyzw(transform: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return quat.astype(np.float64, copy=False)


def matrix_to_xyz_rpy_deg(transform: np.ndarray) -> np.ndarray:
    xyz = np.asarray(transform[:3, 3], dtype=np.float64)
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([xyz, rpy], axis=0)


def load_tum_trajectory(path: Path) -> TrajectoryData:
    timestamps: List[float] = []
    positions: List[np.ndarray] = []
    quats: List[np.ndarray] = []
    transforms: List[np.ndarray] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise TrajectoryValidationError(
                    f"TUM line {line_no} in {path} has fewer than 8 columns."
                )
            try:
                timestamp = float(parts[0])
                position = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
                quat_xyzw = np.asarray([float(v) for v in parts[4:8]], dtype=np.float64)
            except ValueError as exc:
                raise TrajectoryValidationError(
                    f"Failed to parse TUM line {line_no} in {path}: {raw_line.rstrip()}"
                ) from exc

            norm = np.linalg.norm(quat_xyzw)
            if norm < 1e-12:
                raise TrajectoryValidationError(
                    f"Quaternion norm is zero on line {line_no} in {path}."
                )
            quat_xyzw = quat_xyzw / norm

            timestamps.append(timestamp)
            positions.append(position)
            quats.append(quat_xyzw)
            transforms.append(build_transform(position, quat_xyzw))

    if not timestamps:
        raise TrajectoryValidationError(f"No valid TUM poses found in {path}.")

    return TrajectoryData(
        path=path,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        positions_xyz=np.asarray(positions, dtype=np.float64),
        quats_xyzw=np.asarray(quats, dtype=np.float64),
        transforms_world_sensor=np.asarray(transforms, dtype=np.float64),
    )
