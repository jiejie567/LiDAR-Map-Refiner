from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from merge_pcds import PCDCloud, read_pcd, write_pcd
from manual_loop_closure.pcd_io import list_numbered_pcds
from manual_loop_closure.scan_context_io import (
    load_scan_context_gravity,
    load_scan_context_config,
    save_scan_context_database,
)
from manual_loop_closure.trajectory_io import load_tum_trajectory

from .graph_loader import ANCHOR_VERTEX_ID, BetweenFactorRecord, GnssPriorRecord, PosePriorRecord


LogFn = Optional[Callable[[str], None]]


# ===== BEGIN CHANGE: python optimizer exporters =====
@dataclass(frozen=True)
class MeasurementRecord:
    index: int
    odom_time: float
    cloud_path: Path


def _log(log_fn: LogFn, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose.matrix(), dtype=np.float64)


def _make_xyzi_dtype() -> np.dtype:
    return np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
        ]
    )


def _make_xyzi_cloud(points_xyzi: np.ndarray) -> PCDCloud:
    dtype = _make_xyzi_dtype()
    structured = np.empty(points_xyzi.shape[0], dtype=dtype)
    structured["x"] = points_xyzi[:, 0].astype(np.float32, copy=False)
    structured["y"] = points_xyzi[:, 1].astype(np.float32, copy=False)
    structured["z"] = points_xyzi[:, 2].astype(np.float32, copy=False)
    structured["intensity"] = points_xyzi[:, 3].astype(np.float32, copy=False)
    return PCDCloud(
        fields=("x", "y", "z", "intensity"),
        sizes=(4, 4, 4, 4),
        types=("F", "F", "F", "F"),
        counts=(1, 1, 1, 1),
        data_type="binary",
        version="0.7",
        viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        comments=("# .PCD v0.7 - Point Cloud Data file format",),
        data=structured,
        height=1,
    )


def load_xyzi_points(path: Path) -> np.ndarray:
    cloud = read_pcd(path)
    if not {"x", "y", "z"}.issubset(set(cloud.fields)):
        raise RuntimeError(f"PCD is missing x/y/z fields: {path}")
    intensity = (
        cloud.data["intensity"].astype(np.float32, copy=False)
        if "intensity" in cloud.fields
        else np.zeros(cloud.num_points, dtype=np.float32)
    )
    return np.column_stack(
        [
            cloud.data["x"].astype(np.float32, copy=False),
            cloud.data["y"].astype(np.float32, copy=False),
            cloud.data["z"].astype(np.float32, copy=False),
            intensity,
        ]
    )


def _apply_transform(points_xyzi: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = points_xyzi[:, :3].astype(np.float64, copy=False)
    rotated = xyz @ transform[:3, :3].T + transform[:3, 3]
    transformed = np.empty_like(points_xyzi, dtype=np.float32)
    transformed[:, :3] = rotated.astype(np.float32, copy=False)
    transformed[:, 3] = points_xyzi[:, 3].astype(np.float32, copy=False)
    return transformed


def voxel_downsample_xyzi(points_xyzi: np.ndarray, voxel_leaf: float) -> np.ndarray:
    if voxel_leaf <= 0.0 or points_xyzi.size == 0:
        return points_xyzi

    coords = np.floor(points_xyzi[:, :3] / float(voxel_leaf)).astype(np.int64)
    coords = np.ascontiguousarray(coords)
    voxel_keys = coords.view(
        np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    ).reshape(-1)
    _, inverse = np.unique(voxel_keys, return_inverse=True)
    output_count = int(inverse.max()) + 1 if inverse.size else 0
    downsampled = np.zeros((output_count, 4), dtype=np.float64)
    counts = np.bincount(inverse)
    for column in range(4):
        downsampled[:, column] = np.bincount(
            inverse,
            weights=points_xyzi[:, column].astype(np.float64, copy=False),
        ) / counts
    return downsampled.astype(np.float32, copy=False)


def _compact_map_parts(
    parts: list[np.ndarray],
    voxel_leaf: float,
    *,
    log_fn: LogFn = None,
    label: str = "MapExport",
) -> list[np.ndarray]:
    if not parts:
        return []
    merged = np.vstack(parts) if len(parts) > 1 else parts[0]
    filtered = voxel_downsample_xyzi(merged, voxel_leaf)
    _log(
        log_fn,
        f"[{label}] Compacted map points {merged.shape[0]} -> {filtered.shape[0]} "
        f"with voxel={voxel_leaf:.3f}",
    )
    return [filtered]


def save_tum(path: Path, measurements: list[MeasurementRecord], optimized_values, gtsam_mod) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for measurement in measurements:
            pose = optimized_values.atPose3(gtsam_mod.Symbol("x", measurement.index).key())
            matrix = _pose_matrix(pose)
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            stream.write(
                f"{measurement.odom_time:.9f} "
                f"{matrix[0, 3]:.9f} {matrix[1, 3]:.9f} {matrix[2, 3]:.9f} "
                f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
            )


def build_optimized_map(
    measurements: list[MeasurementRecord],
    optimized_values,
    gtsam_mod,
    voxel_leaf: float,
    log_fn: LogFn = None,
) -> np.ndarray:
    merged_parts: list[np.ndarray] = []
    total_frames = len(measurements)
    map_start = time.perf_counter()
    compact_every = 100
    for index, measurement in enumerate(measurements, start=1):
        points_xyzi = load_xyzi_points(measurement.cloud_path)
        pose = optimized_values.atPose3(gtsam_mod.Symbol("x", measurement.index).key())
        transformed = _apply_transform(points_xyzi, _pose_matrix(pose))
        merged_parts.append(voxel_downsample_xyzi(transformed, voxel_leaf))
        if voxel_leaf > 0.0 and index % compact_every == 0:
            merged_parts = _compact_map_parts(
                merged_parts,
                voxel_leaf,
                log_fn=log_fn,
                label="PythonOptimizer",
            )
        if (
            index == 1
            or index == total_frames
            or index % 250 == 0
        ):
            elapsed = time.perf_counter() - map_start
            accumulated_points = sum(part.shape[0] for part in merged_parts)
            _log(
                log_fn,
                "[PythonOptimizer] Map rebuild progress "
                f"{index}/{total_frames} frames, points={accumulated_points}, elapsed={elapsed:.2f}s",
            )
    if not merged_parts:
        return np.empty((0, 4), dtype=np.float32)
    merged = np.vstack(merged_parts)
    filtered = voxel_downsample_xyzi(merged, voxel_leaf)
    elapsed = time.perf_counter() - map_start
    _log(
        log_fn,
        "[PythonOptimizer] Built optimized map "
        f"input_points={merged.shape[0]}, output_points={filtered.shape[0]}, "
        f"voxel={voxel_leaf:.3f}, elapsed={elapsed:.2f}s",
    )
    return filtered


def save_xyzi_pcd(path: Path, points_xyzi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcd(path, _make_xyzi_cloud(points_xyzi))


def colorize_binary_xyzi_pcd(
    source: Path,
    destination: Path,
    color_rgb: tuple[int, int, int],
) -> None:
    """Replace a binary XYZI PCD's scalar field with one constant RGB color."""
    header: list[bytes] = []
    with source.open("rb") as input_stream:
        while True:
            line = input_stream.readline()
            if not line:
                raise RuntimeError(f"PCD DATA header is missing: {source}")
            header.append(line)
            if line.startswith(b"DATA "):
                break
        header_text = b"".join(header).decode("ascii")
        required = (
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            "DATA binary",
        )
        for declaration in required:
            if declaration not in header_text:
                raise RuntimeError(
                    f"unsupported PCD layout ({declaration!r} absent): {source}"
                )

        red, green, blue = (int(value) for value in color_rgb)
        if not all(0 <= value <= 255 for value in (red, green, blue)):
            raise ValueError("RGB components must be in [0, 255]")
        packed_rgb = (red << 16) | (green << 8) | blue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as output_stream:
            output_stream.write(
                header_text.replace(
                    "FIELDS x y z intensity", "FIELDS x y z rgb", 1
                ).encode("ascii")
            )
            while True:
                payload = input_stream.read(16 * 1024 * 1024)
                if not payload:
                    break
                if len(payload) % 16:
                    raise RuntimeError(f"unaligned binary PCD payload: {source}")
                words = np.frombuffer(payload, dtype="<u4").reshape(-1, 4).copy()
                words[:, 3] = packed_rgb
                output_stream.write(words.tobytes())
        temporary.replace(destination)


def save_trajectory_pcd(path: Path, measurements: list[MeasurementRecord], optimized_values, gtsam_mod) -> None:
    points = np.zeros((len(measurements), 4), dtype=np.float32)
    for row, measurement in enumerate(measurements):
        pose = optimized_values.atPose3(gtsam_mod.Symbol("x", measurement.index).key())
        matrix = _pose_matrix(pose)
        points[row, 0:3] = matrix[:3, 3].astype(np.float32, copy=False)
        points[row, 3] = np.float32(measurement.odom_time)
    save_xyzi_pcd(path, points)


def build_map_and_trajectory_from_tum(
    *,
    tum_path: Path,
    keyframe_dir: Path,
    output_map: Path,
    output_trajectory: Path,
    voxel_leaf: float,
    output_scan_context: Path | None = None,
    log_fn: LogFn = None,
) -> tuple[int, int, float]:
    trajectory = load_tum_trajectory(tum_path)
    keyframe_paths = list_numbered_pcds(keyframe_dir)
    if trajectory.size != len(keyframe_paths):
        raise RuntimeError(
            "Keyframe count does not match optimized_poses_tum.txt: "
            f"pcd={len(keyframe_paths)} tum={trajectory.size}"
        )
    config = None
    gravity_up_body = None
    if output_scan_context is not None:
        config = load_scan_context_config(keyframe_dir.parent)
        if config.gravity_canonicalization_enable:
            gravity_up_body = load_scan_context_gravity(keyframe_dir.parent, trajectory)

    merged_parts: list[np.ndarray] = []
    total_frames = trajectory.size
    build_start = time.perf_counter()
    compact_every = 100
    for index, keyframe_path in enumerate(keyframe_paths):
        points_xyzi = load_xyzi_points(keyframe_path)
        transformed = _apply_transform(points_xyzi, trajectory.transforms_world_sensor[index])
        merged_parts.append(voxel_downsample_xyzi(transformed, voxel_leaf))
        if voxel_leaf > 0.0 and (index + 1) % compact_every == 0:
            merged_parts = _compact_map_parts(
                merged_parts,
                voxel_leaf,
                log_fn=log_fn,
                label="MapExport",
            )
        if index == 0 or index + 1 == total_frames or (index + 1) % 250 == 0:
            elapsed = time.perf_counter() - build_start
            accumulated_points = sum(part.shape[0] for part in merged_parts)
            _log(
                log_fn,
                "[MapExport] Progress "
                f"{index + 1}/{total_frames} frames, points={accumulated_points}, elapsed={elapsed:.2f}s",
            )

    merged = np.vstack(merged_parts) if merged_parts else np.empty((0, 4), dtype=np.float32)
    filtered = voxel_downsample_xyzi(merged, voxel_leaf)
    if filtered.size == 0:
        raise RuntimeError("Built map is empty.")

    save_xyzi_pcd(output_map, filtered)

    trajectory_points = np.zeros((trajectory.size, 4), dtype=np.float32)
    trajectory_points[:, 0:3] = trajectory.positions_xyz.astype(np.float32, copy=False)
    trajectory_points[:, 3] = trajectory.timestamps.astype(np.float32, copy=False)
    save_xyzi_pcd(output_trajectory, trajectory_points)

    if output_scan_context is not None:
        assert config is not None
        entry_count = save_scan_context_database(
            path=output_scan_context,
            trajectory=trajectory,
            local_point_sets=(load_xyzi_points(path)[:, :3] for path in keyframe_paths),
            config=config,
            gravity_up_body=gravity_up_body,
        )
        _log(
            log_fn,
            "[MapExport] Wrote corrected Scan Context database "
            f"{output_scan_context} entries={entry_count}",
        )
    _log(
        log_fn,
        "[MapExport] Preserved the complete PGO SE(3) poses; no gravity pose "
        "override or ground alignment was applied.",
    )

    elapsed = time.perf_counter() - build_start
    _log(
        log_fn,
        "[MapExport] Built final map "
        f"input_points={merged.shape[0]}, output_points={filtered.shape[0]}, "
        f"voxel={voxel_leaf:.3f}, elapsed={elapsed:.2f}s",
    )
    return int(filtered.shape[0]), int(trajectory_points.shape[0]), elapsed


def build_scan_context_from_tum(
    *,
    tum_path: Path,
    keyframe_dir: Path,
    output_scan_context: Path,
    log_fn: LogFn = None,
) -> int:
    trajectory = load_tum_trajectory(tum_path)
    keyframe_paths = list_numbered_pcds(keyframe_dir)
    if trajectory.size != len(keyframe_paths):
        raise RuntimeError(
            "Keyframe count does not match optimized_poses_tum.txt: "
            f"pcd={len(keyframe_paths)} tum={trajectory.size}"
        )
    config = load_scan_context_config(keyframe_dir.parent)
    gravity_up_body = (
        load_scan_context_gravity(keyframe_dir.parent, trajectory)
        if config.gravity_canonicalization_enable
        else None
    )
    entry_count = save_scan_context_database(
        path=output_scan_context,
        trajectory=trajectory,
        local_point_sets=(load_xyzi_points(path)[:, :3] for path in keyframe_paths),
        config=config,
        gravity_up_body=gravity_up_body,
    )
    _log(
        log_fn,
        "[MapExport] Wrote Scan Context database with complete PGO poses and "
        f"mapping-time gravity-canonicalized descriptors {output_scan_context} "
        f"entries={entry_count}",
    )
    return entry_count


def _format_upper_triangular_information(information: np.ndarray) -> str:
    values: list[str] = []
    for row in range(6):
        for col in range(row, 6):
            values.append(f"{information[row, col]:.15e}")
    return " ".join(values)


def write_pose_graph_g2o(
    path: Path,
    optimized_values,
    pose_count: int,
    factor_records: list[object],
    gtsam_mod,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for node_id in range(pose_count):
            pose = optimized_values.atPose3(gtsam_mod.Symbol("x", node_id).key())
            matrix = _pose_matrix(pose)
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            stream.write(
                "VERTEX_SE3:QUAT "
                f"{node_id} "
                f"{matrix[0, 3]:.15e} {matrix[1, 3]:.15e} {matrix[2, 3]:.15e} "
                f"{quat[0]:.15e} {quat[1]:.15e} {quat[2]:.15e} {quat[3]:.15e}\n"
            )

        for record in factor_records:
            if not isinstance(record, BetweenFactorRecord):
                continue
            stream.write(
                "EDGE_SE3:QUAT "
                f"{record.node_i} {record.node_j} "
                f"{record.translation_xyz[0]:.15e} {record.translation_xyz[1]:.15e} {record.translation_xyz[2]:.15e} "
                f"{record.quat_xyzw[0]:.15e} {record.quat_xyzw[1]:.15e} {record.quat_xyzw[2]:.15e} {record.quat_xyzw[3]:.15e} "
                f"{_format_upper_triangular_information(record.information)}\n"
            )

        gnss_records = [record for record in factor_records if isinstance(record, GnssPriorRecord)]
        if gnss_records:
            stream.write("# GNSS prior factors serialized by MS-Mapping\n")
            for record in gnss_records:
                if record.subtype == "XYZ":
                    stream.write(
                        "# GNSS_PRIOR XYZ "
                        f"{record.node_id} "
                        f"{record.measurement[0]:.15e} {record.measurement[1]:.15e} {record.measurement[2]:.15e} "
                        f"{record.sigmas.shape[0]} "
                        + " ".join(f"{value:.15e}" for value in record.sigmas)
                        + f" {record.robust_type} {record.robust_param:.15e}\n"
                    )
                elif record.subtype == "POSE":
                    stream.write(
                        "# GNSS_PRIOR POSE "
                        f"{record.node_id} "
                        + " ".join(f"{value:.15e}" for value in record.measurement)
                        + f" {record.sigmas.shape[0]} "
                        + " ".join(f"{value:.15e}" for value in record.sigmas)
                        + f" {record.robust_type} {record.robust_param:.15e}\n"
                    )
                elif record.subtype == "XY":
                    stream.write(
                        "# GNSS_PRIOR XY "
                        f"{record.node_id} "
                        f"{record.measurement[0]:.15e} {record.measurement[1]:.15e} "
                        f"{record.sigmas.shape[0]} "
                        + " ".join(f"{value:.15e}" for value in record.sigmas)
                        + f" {record.robust_type} {record.robust_param:.15e}\n"
                    )

        # Match the legacy C++ exporter: do not serialize the synthetic anchor
        # prior back into the output g2o.


def generate_pose_graph_png(project_root: Path, g2o_path: Path, output_path: Path, log_fn: LogFn = None) -> None:
    candidates = [
        project_root / "gui" / "visualize_pose_graph.py",
        project_root / "scripts" / "visualize_pose_graph.py",
    ]
    script_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if script_path is None:
        raise RuntimeError("visualize_pose_graph.py not found in known repository paths.")
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--g2o",
            str(g2o_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "pose graph png generation failed")
    _log(log_fn, f"[PythonOptimizer] Saved pose graph image: {output_path}")


def save_report_json(
    path: Path,
    *,
    session_root: Path,
    input_g2o: Path,
    input_tum: Path,
    input_keyframe_dir: Path,
    constraints_csv: Path,
    output_dir: Path,
    map_voxel_leaf: float,
    optimize_mode: str,
    total_constraints: int,
    enabled_constraints: int,
    optimized_pose_count: int,
    factor_count: int,
    map_point_count: int,
    map_built: bool,
    map_build_elapsed_sec: float,
    robust_weights: list[float] | None = None,
    manual_factor_weighting: dict | None = None,
) -> None:
    report = {
        "session_root": str(session_root),
        "input_g2o": str(input_g2o),
        "input_tum": str(input_tum),
        "input_keyframe_dir": str(input_keyframe_dir),
        "constraints_csv": str(constraints_csv),
        "output_dir": str(output_dir),
        "map_voxel_leaf": map_voxel_leaf,
        "optimize_mode": optimize_mode,
        "total_constraints": total_constraints,
        "enabled_constraints": enabled_constraints,
        "optimized_pose_count": optimized_pose_count,
        "factor_count": factor_count,
        "map_point_count": map_point_count,
        "map_built": map_built,
        "map_build_elapsed_sec": map_build_elapsed_sec,
        "robust_weights": robust_weights,
        "manual_factor_weighting": manual_factor_weighting,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_report_map_fields(
    path: Path,
    *,
    map_point_count: int,
    map_build_elapsed_sec: float,
) -> None:
    payload: dict
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["map_built"] = True
    payload["map_point_count"] = int(map_point_count)
    payload["map_build_elapsed_sec"] = float(map_build_elapsed_sec)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
# ===== END CHANGE: python optimizer exporters =====
