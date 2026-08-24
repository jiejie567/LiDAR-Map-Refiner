#!/usr/bin/env python3
"""Track fixed real two-sided surface points through guarded/unguarded BALM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from manual_loop_closure.pcd_io import load_xyz_points  # noqa: E402
from manual_loop_closure.trajectory_io import load_tum_trajectory  # noqa: E402
from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
    BalmParams,
    extract_planar_voxels,
    prepare_local_cloud,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clouds(keyframe_dir: Path, count: int, params: BalmParams) -> list[np.ndarray]:
    output = []
    for index in range(count):
        output.append(prepare_local_cloud(load_xyz_points(
            keyframe_dir / f"{index}.pcd"
        ), params))
    return output


def flatten_world(
    clouds: list[np.ndarray], poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.vstack([
        cloud @ pose[:3, :3].T + pose[:3, 3]
        for cloud, pose in zip(clouds, poses)
    ])
    pose_ids = np.repeat(
        np.arange(len(clouds)), [len(cloud) for cloud in clouds]
    )
    local_points = np.vstack(clouds)
    return points, pose_ids, local_points


def project_fixed_rows(
    rows: np.ndarray,
    local_points: np.ndarray,
    pose_ids: np.ndarray,
    poses: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    point_pose = poses[pose_ids[rows]]
    world = np.einsum(
        "nij,nj->ni", point_pose[:, :3, :3], local_points[rows]
    ) + point_pose[:, :3, 3]
    return world @ normal


def fixed_separation(
    candidate: dict,
    local_points: np.ndarray,
    pose_ids: np.ndarray,
    poses: np.ndarray,
) -> float:
    normal = np.asarray(candidate["normal_world"], dtype=np.float64)
    negative = project_fixed_rows(
        candidate["negative_rows"], local_points, pose_ids, poses, normal
    )
    positive = project_fixed_rows(
        candidate["positive_rows"], local_points, pose_ids, poses, normal
    )
    return float(abs(np.median(positive) - np.median(negative)))


def evaluate(
    initial_tum: Path,
    guarded_tum: Path,
    unguarded_tum: Path,
    keyframe_dir: Path,
    output_dir: Path,
    params: BalmParams,
    max_separation_m: float,
    max_candidates: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    initial = load_tum_trajectory(initial_tum)
    guarded = load_tum_trajectory(guarded_tum)
    unguarded = load_tum_trajectory(unguarded_tum)
    if not (initial.size == guarded.size == unguarded.size):
        raise ValueError("trajectory pose counts differ")
    clouds = load_clouds(keyframe_dir, initial.size, params)
    world, pose_ids, local_points = flatten_world(
        clouds, initial.transforms_world_sensor
    )
    diagnostics = []
    extract_planar_voxels(
        world, pose_ids, params,
        initial.transforms_world_sensor[pose_ids, :3, 3],
        diagnostics,
    )
    candidates = [
        item for item in diagnostics
        if params.double_sided_min_separation <= item["separation_m"] <= max_separation_m
        and item["negative_pose_count"] >= params.min_poses_per_voxel
        and item["positive_pose_count"] >= params.min_poses_per_voxel
    ]
    candidates.sort(
        key=lambda item: -(
            len(item["negative_rows"]) + len(item["positive_rows"])
        )
    )
    candidates = candidates[:max_candidates]
    rows = []
    for index, candidate in enumerate(candidates):
        before = fixed_separation(
            candidate, local_points, pose_ids, initial.transforms_world_sensor
        )
        guarded_value = fixed_separation(
            candidate, local_points, pose_ids, guarded.transforms_world_sensor
        )
        unguarded_value = fixed_separation(
            candidate, local_points, pose_ids, unguarded.transforms_world_sensor
        )
        center = candidate["center_world"]
        normal = candidate["normal_world"]
        rows.append({
            "candidate_id": f"DS{index + 1:03d}",
            "layer": candidate["layer"],
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "negative_points": len(candidate["negative_rows"]),
            "positive_points": len(candidate["positive_rows"]),
            "negative_poses": candidate["negative_pose_count"],
            "positive_poses": candidate["positive_pose_count"],
            "initial_separation_m": before,
            "guarded_separation_m": guarded_value,
            "unguarded_separation_m": unguarded_value,
            "guarded_change_m": guarded_value - before,
            "unguarded_change_m": unguarded_value - before,
        })
    csv_path = output_dir / "fixed_double_sided_source_data.csv"
    fields = list(rows[0]) if rows else ["candidate_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixed_identity_policy": (
            "point rows, observation side, layer, normal, and initial candidate "
            "are frozen before either BALM arm"
        ),
        "params": {
            "root_voxel_size": params.root_voxel_size,
            "max_layer": params.max_layer,
            "downsample_leaf": params.downsample_leaf,
            "plane_thickness": params.plane_thickness,
            "double_sided_min_separation": params.double_sided_min_separation,
            "max_candidate_separation_m": max_separation_m,
        },
        "counts": {
            "diagnostic_pairs": len(diagnostics),
            "eligible_pairs": len(candidates),
        },
        "summary": {
            "median_initial_separation_m": float(np.median([
                row["initial_separation_m"] for row in rows
            ])) if rows else None,
            "median_guarded_change_m": float(np.median([
                row["guarded_change_m"] for row in rows
            ])) if rows else None,
            "median_unguarded_change_m": float(np.median([
                row["unguarded_change_m"] for row in rows
            ])) if rows else None,
            "unguarded_thinner_than_guarded_count": sum(
                row["unguarded_separation_m"] < row["guarded_separation_m"]
                for row in rows
            ),
        },
        "inputs": {
            "initial_tum_sha256": sha256_file(initial_tum),
            "guarded_tum_sha256": sha256_file(guarded_tum),
            "unguarded_tum_sha256": sha256_file(unguarded_tum),
        },
        "source_data_sha256": sha256_file(csv_path),
    }
    (output_dir / "fixed_double_sided_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-tum", type=Path, required=True)
    parser.add_argument("--guarded-tum", type=Path, required=True)
    parser.add_argument("--unguarded-tum", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-voxel", type=float, default=4.0)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--downsample-leaf", type=float, default=0.4)
    parser.add_argument("--plane-thickness", type=float, default=0.12)
    parser.add_argument("--min-separation", type=float, default=0.04)
    parser.add_argument("--max-separation", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()
    params = BalmParams(
        root_voxel_size=args.root_voxel,
        max_layer=args.max_layer,
        downsample_leaf=args.downsample_leaf,
        plane_thickness=args.plane_thickness,
        double_sided_enable=True,
        double_sided_min_separation=args.min_separation,
    )
    report = evaluate(
        args.initial_tum.resolve(), args.guarded_tum.resolve(),
        args.unguarded_tum.resolve(), args.keyframe_dir.resolve(),
        args.output_dir.resolve(), params, args.max_separation,
        args.max_candidates,
    )
    print(json.dumps({"counts": report["counts"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
