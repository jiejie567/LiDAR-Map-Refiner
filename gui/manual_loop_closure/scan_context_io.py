from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # package import; leaf-module fallback keeps this usable without open3d
    from manual_loop_closure.trajectory_io import TrajectoryData
except ImportError:  # pragma: no cover
    from trajectory_io import TrajectoryData  # type: ignore


MAGIC = "FAST_LIO_SCAN_CONTEXT_DB_V7"
DEFAULT_SCAN_CONTEXT = {
    "num_rings": 20,
    "num_sectors": 60,
    "max_radius": 80.0,
    "dual_z_layer_enable": False,
    "dual_z_split_height": 2.5,
    "origin_height_from_ground": 0.0,
    "dual_z_low_weight": 0.4,
    "dual_z_high_weight": 0.6,
    "min_joint_rings": 2,
    "retrieval_height_offset": 0.1,
    "sector_support_exponent": 0.5,
    "gravity_canonicalization_enable": True,
}


@dataclass(frozen=True)
class ScanContextConfig:
    num_rings: int = 20
    num_sectors: int = 60
    max_radius: float = 80.0
    dual_z_layer_enable: bool = False
    dual_z_split_height: float = 2.5
    origin_height_from_ground: float = 0.0
    dual_z_low_weight: float = 0.4
    dual_z_high_weight: float = 0.6
    min_joint_rings: int = 2
    retrieval_height_offset: float = 0.1
    sector_support_exponent: float = 0.5
    gravity_canonicalization_enable: bool = True


def _parse_runtime_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    in_scan_context = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "scan_context:":
            in_scan_context = True
            continue
        if not raw_line.startswith((" ", "\t")):
            in_scan_context = False
        if not in_scan_context or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def load_scan_context_config(session_root: Path) -> ScanContextConfig:
    values = _parse_runtime_params(session_root / "runtime_params.yaml")
    low_weight = max(
        0.0,
        float(values.get("dual_z_low_weight", DEFAULT_SCAN_CONTEXT["dual_z_low_weight"])),
    )
    high_weight = max(
        0.0,
        float(values.get("dual_z_high_weight", DEFAULT_SCAN_CONTEXT["dual_z_high_weight"])),
    )
    if low_weight + high_weight <= 1e-12:
        low_weight = float(DEFAULT_SCAN_CONTEXT["dual_z_low_weight"])
        high_weight = float(DEFAULT_SCAN_CONTEXT["dual_z_high_weight"])
    return ScanContextConfig(
        num_rings=max(1, int(values.get("num_rings", DEFAULT_SCAN_CONTEXT["num_rings"]))),
        num_sectors=max(4, int(values.get("num_sectors", DEFAULT_SCAN_CONTEXT["num_sectors"]))),
        max_radius=max(1.0, float(values.get("max_radius", DEFAULT_SCAN_CONTEXT["max_radius"]))),
        dual_z_layer_enable=_parse_bool(
            values.get("dual_z_layer_enable"),
            bool(DEFAULT_SCAN_CONTEXT["dual_z_layer_enable"]),
        ),
        dual_z_split_height=float(
            values.get("dual_z_split_height", DEFAULT_SCAN_CONTEXT["dual_z_split_height"])
        ),
        origin_height_from_ground=max(
            0.0,
            float(
                values.get(
                    "origin_height_from_ground",
                    DEFAULT_SCAN_CONTEXT["origin_height_from_ground"],
                )
            ),
        ),
        dual_z_low_weight=low_weight,
        dual_z_high_weight=high_weight,
        min_joint_rings=max(
            1,
            min(
                max(1, int(values.get("num_rings", DEFAULT_SCAN_CONTEXT["num_rings"]))),
                int(values.get("min_joint_rings", DEFAULT_SCAN_CONTEXT["min_joint_rings"])),
            ),
        ),
        retrieval_height_offset=max(
            0.0,
            float(values.get(
                "retrieval_height_offset",
                DEFAULT_SCAN_CONTEXT["retrieval_height_offset"],
            )),
        ),
        sector_support_exponent=max(
            0.0,
            float(values.get(
                "sector_support_exponent",
                DEFAULT_SCAN_CONTEXT["sector_support_exponent"],
            )),
        ),
        gravity_canonicalization_enable=_parse_bool(
            values.get("gravity_canonicalization_enable"),
            bool(DEFAULT_SCAN_CONTEXT["gravity_canonicalization_enable"]),
        ),
    )


def _gravity_canonical_rotation(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("invalid gravity direction while rebuilding Scan Context")
    source = up / norm
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        # The minimum rotation is ambiguous for an upside-down platform; use a
        # stable 180-degree rotation about +X.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    axis = cross / cross_norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    angle = math.atan2(cross_norm, dot)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def make_descriptor_with_mask(
    points_xyz: np.ndarray,
    config: ScanContextConfig,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = config.num_rings * 2 if config.dual_z_layer_enable else config.num_rings
    descriptor = np.full(
        (row_count, config.num_sectors),
        -1000.0,
        dtype=np.float64,
    )
    valid_cells = np.zeros((row_count, config.num_sectors), dtype=np.bool_)
    if config.dual_z_layer_enable:
        descriptor[config.num_rings :, :] = np.inf

    if points_xyz.size == 0:
        return np.zeros_like(descriptor), valid_cells

    xyz = points_xyz[:, :3].astype(np.float64, copy=False)
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]
    if xyz.size == 0:
        return np.zeros_like(descriptor), valid_cells

    ranges = np.hypot(xyz[:, 0], xyz[:, 1])
    valid = (ranges > 1e-6) & (ranges <= config.max_radius)
    xyz = xyz[valid]
    ranges = ranges[valid]
    if xyz.size == 0:
        return np.zeros_like(descriptor), valid_cells

    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    theta = np.where(theta < 0.0, theta + 2.0 * math.pi, theta)
    ring_idx = np.ceil((ranges / config.max_radius) * config.num_rings).astype(np.int64) - 1
    sector_idx = np.ceil((theta / (2.0 * math.pi)) * config.num_sectors).astype(np.int64) - 1
    ring_idx = np.clip(ring_idx, 0, config.num_rings - 1)
    sector_idx = np.clip(sector_idx, 0, config.num_sectors - 1)
    heights = xyz[:, 2] + config.origin_height_from_ground

    if config.dual_z_layer_enable:
        low_mask = heights <= config.dual_z_split_height
        if np.any(low_mask):
            np.maximum.at(
                descriptor,
                (ring_idx[low_mask], sector_idx[low_mask]),
                heights[low_mask],
            )
            valid_cells[ring_idx[low_mask], sector_idx[low_mask]] = True
        high_mask = ~low_mask
        if np.any(high_mask):
            np.minimum.at(
                descriptor,
                (config.num_rings + ring_idx[high_mask], sector_idx[high_mask]),
                heights[high_mask],
            )
            valid_cells[
                config.num_rings + ring_idx[high_mask], sector_idx[high_mask]
            ] = True
    else:
        np.maximum.at(descriptor, (ring_idx, sector_idx), heights)
        valid_cells[ring_idx, sector_idx] = True

    descriptor[~valid_cells] = 0.0
    return descriptor, valid_cells


def make_descriptor(points_xyz: np.ndarray, config: ScanContextConfig) -> np.ndarray:
    descriptor, _ = make_descriptor_with_mask(points_xyz, config)
    return descriptor


def _matrix_to_rpy(transform: np.ndarray) -> tuple[float, float, float]:
    rotation = transform[:3, :3]
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def load_scan_context_gravity(
    session_dir: Path,
    trajectory: TrajectoryData,
) -> np.ndarray:
    path = session_dir / "scan_context_gravity.csv"
    if not path.is_file():
        raise RuntimeError(
            "Gravity-canonicalized Scan Context export requires the mapping-time "
            f"gravity sidecar: {path}"
        )

    samples: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"index", "stamp", "up_x", "up_y", "up_z"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Invalid gravity sidecar columns: {path}")
        for expected_index, row in enumerate(reader):
            index = int(row["index"])
            stamp = float(row["stamp"])
            up = np.asarray(
                [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])],
                dtype=np.float64,
            )
            if index != expected_index:
                raise RuntimeError(
                    f"Gravity sidecar index mismatch at row {expected_index}: {index}"
                )
            if expected_index >= trajectory.size:
                raise RuntimeError("Gravity sidecar has more entries than the trajectory")
            if abs(stamp - float(trajectory.timestamps[expected_index])) > 1.0e-4:
                raise RuntimeError(
                    f"Gravity sidecar timestamp mismatch at keyframe {index}: "
                    f"gravity={stamp:.9f} trajectory={trajectory.timestamps[index]:.9f}"
                )
            norm = float(np.linalg.norm(up))
            if not np.isfinite(up).all() or norm < 1.0e-9:
                raise RuntimeError(f"Invalid gravity vector at keyframe {index}")
            samples.append(up / norm)

    if len(samples) != trajectory.size:
        raise RuntimeError(
            "Gravity sidecar count mismatch: "
            f"gravity={len(samples)} trajectory={trajectory.size}"
        )
    return np.asarray(samples, dtype=np.float64)


def save_scan_context_database(
    *,
    path: Path,
    trajectory: TrajectoryData,
    local_point_sets: Iterable[np.ndarray],
    config: ScanContextConfig,
    gravity_up_body: np.ndarray | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    point_sets = list(local_point_sets)
    if len(point_sets) != trajectory.size:
        raise RuntimeError(
            "Scan Context keyframe count mismatch: "
            f"pcd={len(point_sets)} tum={trajectory.size}"
        )
    if config.gravity_canonicalization_enable:
        if gravity_up_body is None:
            raise RuntimeError(
                "Gravity-canonicalized Scan Context export requires mapping-time "
                "gravity vectors; optimized poses are not a gravity measurement."
            )
        gravity_up_body = np.asarray(gravity_up_body, dtype=np.float64)
        if gravity_up_body.shape != (trajectory.size, 3):
            raise RuntimeError(
                "Gravity vector count mismatch: "
                f"gravity={gravity_up_body.shape} trajectory={trajectory.size}"
            )

    with path.open("wb") as stream:
        def write_text(value: str) -> None:
            stream.write(value.encode("utf-8"))

        write_text(f"{MAGIC}\n")
        write_text(
            "PARAMS "
            f"{config.num_rings} {config.num_sectors} "
            f"{config.max_radius:.17g} "
            f"{1 if config.dual_z_layer_enable else 0} "
            f"{config.dual_z_split_height:.17g} "
            f"{config.origin_height_from_ground:.17g} "
            f"{config.dual_z_low_weight:.17g} "
            f"{config.dual_z_high_weight:.17g} "
            f"{config.min_joint_rings} "
            f"{1 if config.gravity_canonicalization_enable else 0}\n"
        )
        write_text(f"ENTRIES {trajectory.size}\n")
        for index, points_xyz in enumerate(point_sets):
            pose = trajectory.transforms_world_sensor[index]
            roll, pitch, yaw = _matrix_to_rpy(pose)
            descriptor_points = np.asarray(points_xyz)
            R_g = np.eye(3, dtype=np.float64)
            if config.gravity_canonicalization_enable:
                up_sensor = gravity_up_body[index]
                R_g = _gravity_canonical_rotation(up_sensor)
                descriptor_points = (R_g @ descriptor_points[:, :3].T).T
            R_map_descriptor = pose[:3, :3] @ R_g.T
            canonical_yaw = math.atan2(
                float(R_map_descriptor[1, 0]), float(R_map_descriptor[0, 0])
            )
            write_text(
                "ENTRY "
                f"{index} "
                f"{trajectory.timestamps[index]:.17g} "
                f"{pose[0, 3]:.17g} {pose[1, 3]:.17g} {pose[2, 3]:.17g} "
                f"{roll:.17g} {pitch:.17g} {yaw:.17g} {canonical_yaw:.17g}\n"
            )
            write_text("DESC\n")
            descriptor, valid_cells = make_descriptor_with_mask(descriptor_points, config)
            for row in descriptor:
                write_text(" ".join(f"{value:.17g}" for value in row))
                write_text("\n")
            packed_mask = np.packbits(
                valid_cells.reshape(-1).astype(np.uint8, copy=False),
                bitorder="little",
            )
            write_text(f"MASK_BITS {packed_mask.size}\n")
            stream.write(packed_mask.tobytes())
            write_text("\nEND_ENTRY\n")
    return trajectory.size
