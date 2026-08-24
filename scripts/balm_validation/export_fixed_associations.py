#!/usr/bin/env python3
"""Export deterministic BALM plane sufficient statistics for solver A/B tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
for path in (GUI, GUI / "manual_loop_closure" / "python_optimizer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manual_loop_closure.pcd_io import (  # noqa: E402
    list_numbered_pcds,
    load_xyz_points,
    validate_keyframe_numbering,
)
from manual_loop_closure.trajectory_io import load_tum_trajectory  # noqa: E402
from balm import BalmParams, extract_planar_voxels, prepare_local_cloud  # noqa: E402


MAGIC = b"GBALM2A1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-voxel", type=float, default=4.0)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--downsample-leaf", type=float, default=0.4)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--plane-thickness", type=float, default=0.05)
    parser.add_argument("--min-voxel-points", type=int, default=20)
    parser.add_argument("--min-poses-per-voxel", type=int, default=2)
    parser.add_argument("--max-points-per-voxel-pose", type=int, default=40)
    parser.add_argument("--pose-start", type=int, default=0)
    parser.add_argument("--pose-stop", type=int)
    args = parser.parse_args()

    trajectory = load_tum_trajectory(args.tum.resolve())
    paths = list_numbered_pcds(args.keyframe_dir.resolve())
    validate_keyframe_numbering(paths, trajectory.size)
    stop = trajectory.size if args.pose_stop is None else min(args.pose_stop, trajectory.size)
    if not 0 <= args.pose_start < stop:
        raise ValueError("invalid pose interval")
    selected = np.arange(args.pose_start, stop, dtype=np.int64)
    poses = trajectory.transforms_world_sensor[selected]
    timestamps = trajectory.timestamps[selected]
    params = BalmParams(
        root_voxel_size=args.root_voxel,
        max_layer=args.max_layer,
        downsample_leaf=args.downsample_leaf,
        max_range=args.max_range,
        plane_thickness=args.plane_thickness,
        min_voxel_points=args.min_voxel_points,
        min_poses_per_voxel=args.min_poses_per_voxel,
        max_points_per_voxel_pose=args.max_points_per_voxel_pose,
        double_sided_enable=False,
        pose_prior_translation_weight=0.0,
        pose_prior_rotation_weight=0.0,
    )
    clouds = [
        prepare_local_cloud(load_xyz_points(paths[int(index)]), params)
        for index in selected
    ]
    counts = [len(cloud) for cloud in clouds]
    local = np.vstack(clouds)
    pose_ids = np.repeat(np.arange(len(clouds)), counts)
    world = np.einsum(
        "mi,mji->mj", local, poses[pose_ids, :3, :3]
    ) + poses[pose_ids, :3, 3]
    plane_rows = extract_planar_voxels(world, pose_ids, params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<III", 1, len(poses), len(plane_rows)))
        for timestamp, pose in zip(timestamps, poses):
            stream.write(struct.pack("<d", float(timestamp)))
            stream.write(np.asarray(pose[:3, :3], dtype="<f8").tobytes(order="C"))
            stream.write(np.asarray(pose[:3, 3], dtype="<f8").tobytes(order="C"))
        for rows in plane_rows:
            ids = pose_ids[rows]
            unique, starts = np.unique(ids, return_index=True)
            stream.write(struct.pack("<I", len(unique)))
            for ordinal, pose_id in enumerate(unique):
                begin = starts[ordinal]
                end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(rows)
                points = local[rows[begin:end]]
                count = len(points)
                first = points.sum(axis=0, dtype=np.float64)
                second = points.T @ points
                stream.write(struct.pack("<II", int(pose_id), count))
                stream.write(np.asarray(first, dtype="<f8").tobytes(order="C"))
                stream.write(np.asarray(second, dtype="<f8").tobytes(order="C"))

    report = {
        "schema_version": 1,
        "format": "GBALM2A1 little-endian sufficient statistics",
        "input_tum": str(args.tum.resolve()),
        "input_tum_sha256": sha256_file(args.tum.resolve()),
        "keyframe_dir": str(args.keyframe_dir.resolve()),
        "pose_interval": [int(args.pose_start), int(stop)],
        "pose_count": len(poses),
        "downsampled_point_count": int(len(local)),
        "plane_count": len(plane_rows),
        "associated_point_count": int(sum(len(rows) for rows in plane_rows)),
        "params": {
            key: getattr(params, key)
            for key in (
                "root_voxel_size", "max_layer", "downsample_leaf", "max_range",
                "plane_thickness", "min_voxel_points", "min_poses_per_voxel",
                "max_points_per_voxel_pose",
            )
        },
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
