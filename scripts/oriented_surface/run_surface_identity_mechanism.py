#!/usr/bin/env python3
"""Controlled tests of observation-side surface identity in registration and BA.

The experiment has two deliberately different failure mechanisms:

1. a scan of one partition face is initialized closer to the opposite face in
   a two-face target submap; and
2. four keyframes observe the two physical faces of a thin partition, whose
   combined samples satisfy a conventional voxel-plane test.

The first measures wrong-face capture during point-to-plane registration.  The
second measures wall-thickness bias during BALM.  Both arms share every random
draw, point sample, initial pose, and numerical setting except the explicit
observation-side identity switch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
    BalmParams,
    run_balm_refinement,
)
from manual_loop_closure.python_optimizer.oriented_surface_factors import (  # noqa: E402
    OrientedFactorConfig,
    estimate_oriented_surface_factor,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_pose(yaw_deg: float, xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "z", yaw_deg, degrees=True
    ).as_matrix()
    pose[:3, 3] = xyz
    return pose


def registration_trial(
    thickness_m: float,
    seed: int,
    point_count: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    yz = np.column_stack([
        rng.uniform(-3.0, 3.0, point_count),
        rng.uniform(-1.5, 1.5, point_count),
    ])
    source = np.column_stack([
        rng.normal(0.0, 0.0015, point_count), yz
    ])
    same = np.column_stack([
        rng.normal(0.0, 0.0015, point_count),
        yz + rng.normal(0.0, 0.001, yz.shape),
    ])
    opposite = np.column_stack([
        rng.normal(thickness_m, 0.0015, point_count),
        yz + rng.normal(0.0, 0.001, yz.shape),
    ])
    target = np.vstack([same, opposite])
    source_normals = np.tile([-1.0, 0.0, 0.0], (point_count, 1))
    target_normals = np.vstack([
        np.tile([-1.0, 0.0, 0.0], (point_count, 1)),
        np.tile([1.0, 0.0, 0.0], (point_count, 1)),
    ])
    initial = np.eye(4, dtype=np.float64)
    initial_fraction = float(rng.uniform(0.58, 0.92))
    initial[0, 3] = initial_fraction * thickness_m
    common = dict(
        voxel_size_m=0.0,
        normal_radius_m=0.2,
        max_correspondence_m=max(0.30, 2.0 * thickness_m),
        max_iterations=20,
        min_correspondences=max(80, point_count // 4),
        min_overlap=0.7,
        huber_delta_m=0.08,
    )
    configurations = {
        "spatial_nn": OrientedFactorConfig(
            **common, normal_gate_deg=180.0, normal_search_k=1,
        ),
        "observation_side": OrientedFactorConfig(
            **common, normal_gate_deg=30.0, normal_search_k=8,
        ),
    }
    rows = []
    for method, config in configurations.items():
        result = estimate_oriented_surface_factor(
            source, target, initial, config,
            source_normals_local=source_normals,
            target_normals_local=target_normals,
        )
        estimate_x = float(result.transform_target_source[0, 3])
        true_error = abs(estimate_x)
        opposite_error = abs(estimate_x - thickness_m)
        rows.append({
            "experiment": "registration",
            "method": method,
            "seed": seed,
            "thickness_m": thickness_m,
            "initial_fraction": initial_fraction,
            "initial_x_m": float(initial[0, 3]),
            "valid": int(result.valid),
            "reason": result.reason,
            "estimate_x_m": estimate_x,
            "true_face_error_m": true_error,
            "opposite_face_error_m": opposite_error,
            "correct_face": int(result.valid and true_error < thickness_m / 4.0),
            "wrong_face": int(result.valid and opposite_error < thickness_m / 4.0),
            "overlap": result.overlap,
            "spatial_overlap": result.spatial_overlap,
            "inlier_rmse_m": result.inlier_rmse_m,
            "iterations": result.iterations,
        })
    return rows


def build_ba_scene(
    thickness_m: float,
    seed: int,
    point_count: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)

    def sheet(x_position: float) -> np.ndarray:
        return np.column_stack([
            rng.normal(x_position, 0.004, point_count),
            rng.uniform(0.0, 6.0, point_count),
            rng.uniform(0.6, 3.0, point_count),
        ])

    def floor(x_low: float, x_high: float) -> np.ndarray:
        return np.column_stack([
            rng.uniform(x_low, x_high, point_count),
            rng.uniform(0.0, 6.0, point_count),
            rng.normal(0.5, 0.004, point_count),
        ])

    # Keep both faces inside the same 1 m root voxel.  Centering the partition
    # on a voxel boundary would trivially separate them before either method's
    # association policy is exercised.
    partition_center_x = 0.5
    positive = sheet(partition_center_x + 0.5 * thickness_m)
    negative = sheet(partition_center_x - 0.5 * thickness_m)
    poses = np.stack([
        make_pose(180.0, (2.5, 1.4, 1.5)),
        make_pose(155.0, (2.5, 4.6, 1.5)),
        make_pose(0.0, (-2.5, 1.4, 1.5)),
        make_pose(-25.0, (-2.5, 4.6, 1.5)),
    ])
    visible = [
        [positive, floor(0.8, 4.2)],
        [positive, floor(0.8, 4.2)],
        [negative, floor(-3.2, 0.2)],
        [negative, floor(-3.2, 0.2)],
    ]
    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for pose, planes in zip(poses, visible):
        world = np.vstack(planes)
        clouds.append((world - pose[:3, 3]) @ pose[:3, :3])
        label = np.zeros(len(world), dtype=np.int8)
        label[:point_count] = 1 if planes[0] is positive else -1
        labels.append(label)
    perturbed = poses.copy()
    # Pose 0 is the common gauge and is deliberately identical in both arms.
    for index in range(1, len(perturbed)):
        perturbed[index, :3, :3] = (
            perturbed[index, :3, :3]
            @ Rotation.from_rotvec(rng.normal(0.0, 0.008, 3)).as_matrix()
        )
        perturbed[index, :3, 3] += rng.normal(0.0, 0.025, 3)
    return poses, clouds, labels, perturbed


def wall_thickness(
    poses: np.ndarray,
    clouds: list[np.ndarray],
    labels: list[np.ndarray],
) -> float:
    faces: dict[int, list[np.ndarray]] = {1: [], -1: []}
    for pose, cloud, label in zip(poses, clouds, labels):
        world = cloud @ pose[:3, :3].T + pose[:3, 3]
        for face in (1, -1):
            if np.any(label == face):
                faces[face].append(world[label == face, 0])
    return float(
        np.median(np.concatenate(faces[1]))
        - np.median(np.concatenate(faces[-1]))
    )


def pose_errors(
    truth: np.ndarray, estimate: np.ndarray,
) -> tuple[float, float]:
    translation = np.linalg.norm(
        estimate[:, :3, 3] - truth[:, :3, 3], axis=1
    )
    rotation = []
    for reference, candidate in zip(truth, estimate):
        rotation.append(Rotation.from_matrix(
            reference[:3, :3].T @ candidate[:3, :3]
        ).magnitude())
    return (
        float(np.sqrt(np.mean(np.square(translation)))),
        float(np.degrees(np.sqrt(np.mean(np.square(rotation))))),
    )


def ba_trial(
    thickness_m: float,
    seed: int,
    point_count: int,
) -> list[dict]:
    truth, clouds, labels, initial = build_ba_scene(
        thickness_m, seed, point_count
    )
    initial_thickness = wall_thickness(initial, clouds, labels)
    initial_trans, initial_rot = pose_errors(truth, initial)
    common = dict(
        root_voxel_size=1.0,
        max_layer=3,
        min_voxel_points=20,
        min_poses_per_voxel=2,
        plane_thickness=0.12,
        downsample_leaf=0.0,
        max_points_per_voxel_pose=40,
        max_iterations=12,
        reassociate_every=3,
        lm_damping=1e-1,
        robust_kernel="huber",
        double_sided_min_separation=0.025,
        pose_prior_translation_weight=300.0,
        pose_prior_rotation_weight=1200.0,
    )
    rows = []
    for method, enabled in (("voxel_plane", False), ("observation_side", True)):
        params = BalmParams(**common, double_sided_enable=enabled)
        result = run_balm_refinement(clouds, initial, params)
        final_thickness = wall_thickness(
            result.poses_world_sensor, clouds, labels
        )
        trans_rmse, rot_rmse = pose_errors(
            truth, result.poses_world_sensor
        )
        rows.append({
            "experiment": "bundle_adjustment",
            "method": method,
            "seed": seed,
            "thickness_m": thickness_m,
            "initial_thickness_m": initial_thickness,
            "final_thickness_m": final_thickness,
            "absolute_thickness_error_m": abs(final_thickness - thickness_m),
            "relative_thickness_error": abs(final_thickness - thickness_m) / thickness_m,
            "thinning_m": initial_thickness - final_thickness,
            "initial_translation_rmse_m": initial_trans,
            "final_translation_rmse_m": trans_rmse,
            "initial_rotation_rmse_deg": initial_rot,
            "final_rotation_rmse_deg": rot_rmse,
            "iterations": len(result.iterations),
            "stop_reason": result.stop_reason,
            "final_plane_rms_m": (
                result.iterations[-1].rms_plane_distance
                if result.iterations else float("nan")
            ),
            "elapsed_sec": result.elapsed_sec,
        })
    return rows


def median(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.median(values)) if values else float("nan")


def summarize_registration(rows: list[dict]) -> dict:
    output = {}
    for method in ("spatial_nn", "observation_side"):
        selected = [row for row in rows if row["method"] == method]
        output[method] = {
            "trials": len(selected),
            "valid_rate": float(np.mean([row["valid"] for row in selected])),
            "correct_face_rate": float(np.mean([
                row["correct_face"] for row in selected
            ])),
            "wrong_face_rate": float(np.mean([
                row["wrong_face"] for row in selected
            ])),
            "median_true_face_error_m": median(selected, "true_face_error_m"),
        }
    return output


def summarize_ba(rows: list[dict]) -> dict:
    output = {}
    for method in ("voxel_plane", "observation_side"):
        selected = [row for row in rows if row["method"] == method]
        output[method] = {
            "trials": len(selected),
            "median_absolute_thickness_error_m": median(
                selected, "absolute_thickness_error_m"
            ),
            "median_relative_thickness_error": median(
                selected, "relative_thickness_error"
            ),
            "median_thinning_m": median(selected, "thinning_m"),
            "median_translation_rmse_m": median(
                selected, "final_translation_rmse_m"
            ),
            "median_rotation_rmse_deg": median(
                selected, "final_rotation_rmse_deg"
            ),
            "median_final_plane_rms_m": median(selected, "final_plane_rms_m"),
            "median_elapsed_sec": median(selected, "elapsed_sec"),
        }
    return output


def run(
    output_dir: Path,
    thicknesses: list[float],
    seeds: int,
    registration_points: int,
    ba_points: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    registration_rows: list[dict] = []
    ba_rows: list[dict] = []
    for thickness_index, thickness in enumerate(thicknesses):
        for trial in range(seeds):
            seed = 10000 * thickness_index + trial
            registration_rows.extend(registration_trial(
                thickness, seed, registration_points
            ))
            ba_rows.extend(ba_trial(thickness, seed, ba_points))
    registration_csv = output_dir / "registration_source_data.csv"
    ba_csv = output_dir / "bundle_adjustment_source_data.csv"
    write_csv(registration_csv, registration_rows)
    write_csv(ba_csv, ba_rows)
    registration_summary = summarize_registration(registration_rows)
    ba_summary = summarize_ba(ba_rows)
    baseline_error = ba_summary["voxel_plane"][
        "median_absolute_thickness_error_m"
    ]
    oriented_error = ba_summary["observation_side"][
        "median_absolute_thickness_error_m"
    ]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": (
            "observation-side identity prevents wrong-face registration and "
            "voxel-plane thinning without using ground truth in the method"
        ),
        "control_policy": (
            "paired seeds, points, poses, initialization, robust kernels, and "
            "iteration budgets; only surface-identity association differs"
        ),
        "parameters": {
            "thicknesses_m": thicknesses,
            "seeds_per_thickness": seeds,
            "registration_points_per_face": registration_points,
            "ba_points_per_surface": ba_points,
            "registration_config": asdict(OrientedFactorConfig()),
        },
        "registration": registration_summary,
        "bundle_adjustment": ba_summary,
        "gates": {
            "registration_correct_face_rate_at_least_0_90": (
                registration_summary["observation_side"]["correct_face_rate"] >= 0.90
            ),
            "baseline_wrong_face_rate_at_least_0_50": (
                registration_summary["spatial_nn"]["wrong_face_rate"] >= 0.50
            ),
            "thickness_error_reduction_at_least_30_percent": (
                baseline_error > 0.0
                and oriented_error <= 0.70 * baseline_error
            ),
            "translation_rmse_penalty_below_2_mm": (
                ba_summary["observation_side"]["median_translation_rmse_m"]
                <= ba_summary["voxel_plane"]["median_translation_rmse_m"] + 0.002
            ),
        },
        "inputs": {
            "script_sha256": sha256_file(HERE),
            "factor_builder_sha256": sha256_file(
                GUI / "manual_loop_closure" / "python_optimizer"
                / "oriented_surface_factors.py"
            ),
            "balm_sha256": sha256_file(
                GUI / "manual_loop_closure" / "python_optimizer" / "balm.py"
            ),
        },
        "source_data": {
            "registration_csv": registration_csv.name,
            "registration_csv_sha256": sha256_file(registration_csv),
            "bundle_adjustment_csv": ba_csv.name,
            "bundle_adjustment_csv_sha256": sha256_file(ba_csv),
        },
    }
    report["all_gates_pass"] = all(report["gates"].values())
    report_path = output_dir / "mechanism_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thicknesses", type=float, nargs="+",
        default=[0.06, 0.08, 0.12, 0.16, 0.24],
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--registration-points", type=int, default=500)
    parser.add_argument("--ba-points", type=int, default=350)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if any(value <= 0.0 for value in args.thicknesses):
        parser.error("all thicknesses must be positive")
    report = run(
        args.output_dir.resolve(), args.thicknesses, args.seeds,
        args.registration_points, args.ba_points,
    )
    print(json.dumps({
        "registration": report["registration"],
        "bundle_adjustment": report["bundle_adjustment"],
        "gates": report["gates"],
        "all_gates_pass": report["all_gates_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
