#!/usr/bin/env python3
"""Generate deterministic odometry-proximity loop proposals for factor ablation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrate_factor_information import tum_data
from rebuild_oriented_pose_graph import sha256_file


CSV_FIELDS = [
    "enabled", "source_id", "target_id", "tx", "ty", "tz",
    "qx", "qy", "qz", "qw", "sigma_tx", "sigma_ty", "sigma_tz",
    "sigma_roll_deg", "sigma_pitch_deg", "sigma_yaw_deg",
]


def proximity_pairs(
    timestamps: np.ndarray,
    world_poses: np.ndarray,
    *,
    radius_m: float,
    min_time_gap_s: float,
    max_candidates: int,
    suppression_frames: int,
) -> list[tuple[int, int, float]]:
    positions = np.asarray(world_poses[:, :3, 3], dtype=np.float64)
    tree = cKDTree(positions)
    candidates = []
    for target_id, neighbors in enumerate(tree.query_ball_point(positions, radius_m)):
        for source_id in neighbors:
            if source_id <= target_id:
                continue
            if timestamps[source_id] - timestamps[target_id] < min_time_gap_s:
                continue
            distance = float(np.linalg.norm(positions[source_id] - positions[target_id]))
            candidates.append((target_id, source_id, distance))
    candidates.sort(key=lambda item: (item[2], item[0], item[1]))
    selected = []
    for candidate in candidates:
        target_id, source_id, _distance = candidate
        if any(
            abs(target_id - old_target) <= suppression_frames
            and abs(source_id - old_source) <= suppression_frames
            for old_target, old_source, _ in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def loop_row(
    target_id: int, source_id: int, world_poses: np.ndarray,
    sigma_translation_m: float, sigma_rotation_deg: float,
) -> dict:
    target_source = np.linalg.inv(world_poses[target_id]) @ world_poses[source_id]
    quaternion = Rotation.from_matrix(target_source[:3, :3]).as_quat()
    return {
        "enabled": 1,
        "source_id": source_id,
        "target_id": target_id,
        "tx": float(target_source[0, 3]),
        "ty": float(target_source[1, 3]),
        "tz": float(target_source[2, 3]),
        "qx": float(quaternion[0]),
        "qy": float(quaternion[1]),
        "qz": float(quaternion[2]),
        "qw": float(quaternion[3]),
        "sigma_tx": sigma_translation_m,
        "sigma_ty": sigma_translation_m,
        "sigma_tz": sigma_translation_m,
        "sigma_roll_deg": sigma_rotation_deg,
        "sigma_pitch_deg": sigma_rotation_deg,
        "sigma_yaw_deg": sigma_rotation_deg,
    }


def generate(
    session: Path,
    output_dir: Path,
    *,
    radius_m: float,
    min_time_gap_s: float,
    max_candidates: int,
    suppression_frames: int,
    sigma_translation_m: float,
    sigma_rotation_deg: float,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    tum_path = session / "optimized_poses_tum.txt"
    timestamps, poses = tum_data(tum_path)
    pairs = proximity_pairs(
        timestamps, poses, radius_m=radius_m,
        min_time_gap_s=min_time_gap_s, max_candidates=max_candidates,
        suppression_frames=suppression_frames,
    )
    rows = [
        loop_row(target_id, source_id, poses, sigma_translation_m, sigma_rotation_deg)
        for target_id, source_id, _ in pairs
    ]
    csv_path = output_dir / "loop_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    source_data = output_dir / "loop_candidate_source_data.csv"
    with source_data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["target_id", "source_id", "odometry_distance_m",
                        "time_gap_s"],
        )
        writer.writeheader()
        for target_id, source_id, distance in pairs:
            writer.writerow({
                "target_id": target_id,
                "source_id": source_id,
                "odometry_distance_m": distance,
                "time_gap_s": float(timestamps[source_id] - timestamps[target_id]),
            })
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "proposal_source": "odometry_position_proximity",
        "measurement_initialization": "relative_pose_from_input_odometry",
        "session": str(session.resolve()),
        "input_tum_sha256": sha256_file(tum_path),
        "parameters": {
            "radius_m": radius_m,
            "min_time_gap_s": min_time_gap_s,
            "max_candidates": max_candidates,
            "suppression_frames": suppression_frames,
            "sigma_translation_m": sigma_translation_m,
            "sigma_rotation_deg": sigma_rotation_deg,
        },
        "candidate_count": len(rows),
        "outputs": {
            "loop_candidates_sha256": sha256_file(csv_path),
            "source_data_sha256": sha256_file(source_data),
        },
    }
    (output_dir / "loop_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--min-time-gap", type=float, default=30.0)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--suppression-frames", type=int, default=20)
    parser.add_argument("--sigma-translation", type=float, default=1.0)
    parser.add_argument("--sigma-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()
    report = generate(
        args.session.expanduser().resolve(), args.output_dir.expanduser().resolve(),
        radius_m=args.radius, min_time_gap_s=args.min_time_gap,
        max_candidates=args.max_candidates,
        suppression_frames=args.suppression_frames,
        sigma_translation_m=args.sigma_translation,
        sigma_rotation_deg=args.sigma_rotation_deg,
    )
    print(json.dumps({"candidate_count": report["candidate_count"]}, indent=2))


if __name__ == "__main__":
    main()
