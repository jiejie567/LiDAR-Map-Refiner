#!/usr/bin/env python3
"""Run GhostLoop's joint solver once on frozen initial plane associations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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
from balm import BalmParams, prepare_local_cloud, run_balm_refinement  # noqa: E402


def write_tum(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with path.open("w", encoding="utf-8") as stream:
        for timestamp, pose, quat in zip(timestamps, poses, quaternion):
            stream.write(
                f"{timestamp:.9f} {pose[0, 3]:.9f} {pose[1, 3]:.9f} "
                f"{pose[2, 3]:.9f} {quat[0]:.9f} {quat[1]:.9f} "
                f"{quat[2]:.9f} {quat[3]:.9f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--output-tum", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root-voxel", type=float, default=4.0)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--downsample-leaf", type=float, default=0.4)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--plane-thickness", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--pose-start", type=int, default=0)
    parser.add_argument("--pose-stop", type=int)
    args = parser.parse_args()

    trajectory = load_tum_trajectory(args.tum.resolve())
    paths = list_numbered_pcds(args.keyframe_dir.resolve())
    validate_keyframe_numbering(paths, trajectory.size)
    stop = trajectory.size if args.pose_stop is None else min(args.pose_stop, trajectory.size)
    selected = np.arange(args.pose_start, stop, dtype=np.int64)
    params = BalmParams(
        root_voxel_size=args.root_voxel,
        max_layer=args.max_layer,
        downsample_leaf=args.downsample_leaf,
        max_range=args.max_range,
        plane_thickness=args.plane_thickness,
        max_iterations=args.iterations,
        reassociate_every=0,
        lm_damping=args.damping,
        robust_kernel="none",
        pose_prior_translation_weight=0.0,
        pose_prior_rotation_weight=0.0,
        double_sided_enable=False,
    )
    clouds = [
        prepare_local_cloud(load_xyz_points(paths[int(index)]), params)
        for index in selected
    ]
    result = run_balm_refinement(
        clouds,
        trajectory.transforms_world_sensor[selected],
        params,
        log_fn=print,
    )
    args.output_tum.parent.mkdir(parents=True, exist_ok=True)
    write_tum(
        args.output_tum,
        trajectory.timestamps[selected],
        result.poses_world_sensor,
    )
    report = {
        "solver": "joint_plane_schur_lm_fixed_associations",
        "pose_interval": [int(args.pose_start), int(stop)],
        "params": {
            "root_voxel_size": params.root_voxel_size,
            "max_layer": params.max_layer,
            "downsample_leaf": params.downsample_leaf,
            "max_range": params.max_range,
            "plane_thickness": params.plane_thickness,
            "max_iterations": params.max_iterations,
            "initial_lm_damping": params.lm_damping,
            "robust_kernel": params.robust_kernel,
            "reassociate_every": params.reassociate_every,
            "pose_prior_translation_weight": params.pose_prior_translation_weight,
            "pose_prior_rotation_weight": params.pose_prior_rotation_weight,
        },
        **result.report_dict(),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
