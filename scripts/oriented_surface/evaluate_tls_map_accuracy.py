#!/usr/bin/env python3
"""Evaluate fixed mobile-LiDAR points against a survey-grade TLS map.

The evaluator applies one rigid alignment estimated from the *input* PGO
trajectory to all compared BA arms.  It never independently realigns a method,
and it reprojects exactly the same local point identities for every arm.  In
addition to whole-map accuracy, it freezes the initial observation-side
double-surface candidates and reports their paired TLS distances separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
GHOST_EVAL = HERE.parents[1] / "ghostloop_eval"
for extra in (GUI, GHOST_EVAL):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_icra2027_rebuild import (  # noqa: E402
    interpolated_truth_in_keyframe_frame,
)
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


def read_binary_pcd_xyz_stride(path: Path, stride: int) -> np.ndarray:
    """Memory-map a simple binary PCD and deterministically subsample rows."""
    metadata: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"PCD header is incomplete: {path}")
            line = raw.decode("ascii", errors="ignore").strip()
            parts = line.split()
            if parts and not line.startswith("#"):
                metadata[parts[0].upper()] = parts[1:]
            if parts and parts[0].upper() == "DATA":
                if len(parts) != 2 or parts[1].lower() != "binary":
                    raise ValueError("TLS evaluator currently requires binary PCD")
                offset = stream.tell()
                break
    fields = metadata["FIELDS"]
    sizes = [int(value) for value in metadata["SIZE"]]
    types = metadata["TYPE"]
    counts = [int(value) for value in metadata.get(
        "COUNT", ["1"] * len(fields)
    )]
    if any(value != 1 for value in counts):
        raise ValueError("vector-valued PCD fields are not supported")
    type_map = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
    }
    dtype = np.dtype([
        (name, type_map[(kind.upper(), size)])
        for name, kind, size in zip(fields, types, sizes)
    ])
    point_count = int(metadata.get("POINTS", metadata["WIDTH"])[0])
    mapped = np.memmap(
        path, dtype=dtype, mode="r", offset=offset, shape=(point_count,)
    )
    rows = slice(None, None, max(1, int(stride)))
    xyz = np.column_stack([
        mapped["x"][rows], mapped["y"][rows], mapped["z"][rows]
    ]).astype(np.float64, copy=False)
    return xyz[np.isfinite(xyz).all(axis=1)]


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _singular, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def aligned_poses(
    poses: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
) -> np.ndarray:
    output = poses.copy()
    output[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    output[:, :3, 3] = poses[:, :3, 3] @ rotation.T + translation
    return output


def load_fixed_points(
    keyframe_dir: Path, pose_count: int, params: BalmParams,
) -> tuple[np.ndarray, np.ndarray]:
    local = []
    counts = []
    for index in range(pose_count):
        cloud = prepare_local_cloud(
            load_xyz_points(keyframe_dir / f"{index}.pcd"), params
        )
        local.append(cloud)
        counts.append(len(cloud))
    return np.vstack(local), np.repeat(np.arange(pose_count), counts)


def project_rows(
    rows: np.ndarray,
    local_points: np.ndarray,
    pose_ids: np.ndarray,
    poses: np.ndarray,
) -> np.ndarray:
    selected_pose = poses[pose_ids[rows]]
    return np.einsum(
        "nij,nj->ni", selected_pose[:, :3, :3], local_points[rows]
    ) + selected_pose[:, :3, 3]


def query_distances(
    tree: cKDTree,
    points: np.ndarray,
    max_distance_m: float,
    chunk_size: int = 200_000,
) -> np.ndarray:
    output = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        stop = min(len(points), start + chunk_size)
        values, _ = tree.query(
            points[start:stop], k=1,
            distance_upper_bound=max_distance_m, workers=-1,
        )
        output[start:stop] = values.astype(np.float32)
    return output


def query_nearest(
    tree: cKDTree,
    points: np.ndarray,
    max_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance, index = tree.query(
        points, k=1, distance_upper_bound=max_distance_m, workers=-1
    )
    return distance, index


def projected_median_and_mad(
    points: np.ndarray, normal: np.ndarray,
) -> tuple[float, float]:
    projection = points @ normal
    center = float(np.median(projection))
    mad = float(np.median(np.abs(projection - center)))
    return center, mad


def distance_stats(distances: np.ndarray, cap_m: float) -> dict:
    finite = np.isfinite(distances)
    clipped = np.minimum(
        np.where(finite, distances, cap_m), cap_m
    ).astype(np.float64)
    return {
        "point_count": int(len(distances)),
        "matched_within_cap_rate": float(finite.mean()),
        "median_clipped_m": float(np.median(clipped)),
        "p90_clipped_m": float(np.percentile(clipped, 90)),
        "truncated_rmse_m": float(np.sqrt(np.mean(np.square(clipped)))),
        "within_0_05m_rate": float(np.mean(distances <= 0.05)),
        "within_0_10m_rate": float(np.mean(distances <= 0.10)),
        "within_0_20m_rate": float(np.mean(distances <= 0.20)),
        "within_0_50m_rate": float(np.mean(distances <= 0.50)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    trajectories = {
        "input_pgo": load_tum_trajectory(args.input_tum),
        "side_aware": load_tum_trajectory(args.side_aware_tum),
        "unoriented": load_tum_trajectory(args.unoriented_tum),
    }
    pose_count = trajectories["input_pgo"].size
    if any(item.size != pose_count for item in trajectories.values()):
        raise ValueError("trajectory pose counts differ")
    frame_config = json.loads(args.frame_config.read_text(encoding="utf-8"))
    truth_poses, truth_valid, _ = interpolated_truth_in_keyframe_frame(
        trajectories["input_pgo"].timestamps,
        args.truth_tum, args.dataset, frame_config,
    )
    estimate_positions = trajectories[
        "input_pgo"
    ].transforms_world_sensor[truth_valid, :3, 3]
    truth_positions = truth_poses[truth_valid, :3, 3]
    alignment_rotation, alignment_translation = rigid_alignment(
        estimate_positions, truth_positions
    )
    aligned = {
        name: aligned_poses(
            item.transforms_world_sensor,
            alignment_rotation, alignment_translation,
        )
        for name, item in trajectories.items()
    }

    params = BalmParams(
        root_voxel_size=args.root_voxel,
        max_layer=args.max_layer,
        downsample_leaf=args.downsample_leaf,
        max_range=args.max_observation_range,
        plane_thickness=args.plane_thickness,
        double_sided_enable=True,
        double_sided_min_separation=args.min_separation,
    )
    local_points, pose_ids = load_fixed_points(
        args.keyframe_dir, pose_count, params
    )
    tls_points = read_binary_pcd_xyz_stride(args.tls_map, args.tls_stride)
    tls_tree = cKDTree(tls_points)

    all_rows = np.arange(len(local_points), dtype=np.int64)
    global_stats = {}
    global_distances: dict[str, np.ndarray] = {}
    for name, poses in aligned.items():
        distances = query_distances(
            tls_tree,
            project_rows(all_rows, local_points, pose_ids, poses),
            args.distance_cap,
        )
        global_distances[name] = distances
        global_stats[name] = distance_stats(distances, args.distance_cap)

    initial_world = project_rows(
        all_rows, local_points, pose_ids, aligned["input_pgo"]
    )
    diagnostics: list[dict] = []
    extract_planar_voxels(
        initial_world, pose_ids, params,
        aligned["input_pgo"][pose_ids, :3, 3], diagnostics,
    )
    candidates = [
        item for item in diagnostics
        if args.min_separation <= item["separation_m"] <= args.max_separation
        and abs(float(item["normal_world"][2])) < args.max_abs_normal_z
        and item["negative_pose_count"] >= params.min_poses_per_voxel
        and item["positive_pose_count"] >= params.min_poses_per_voxel
    ]
    candidates.sort(key=lambda item: -(
        len(item["negative_rows"]) + len(item["positive_rows"])
    ))
    candidate_rows = []
    for index, candidate in enumerate(candidates):
        negative_rows = np.asarray(candidate["negative_rows"], dtype=np.int64)
        positive_rows = np.asarray(candidate["positive_rows"], dtype=np.int64)
        rows = np.unique(np.concatenate([negative_rows, positive_rows]))
        normal = np.asarray(candidate["normal_world"], dtype=np.float64)
        row = {
            "candidate_id": f"TLS{index + 1:03d}",
            "layer": int(candidate["layer"]),
            "center_x": float(candidate["center_world"][0]),
            "center_y": float(candidate["center_world"][1]),
            "center_z": float(candidate["center_world"][2]),
            "normal_x": float(candidate["normal_world"][0]),
            "normal_y": float(candidate["normal_world"][1]),
            "normal_z": float(candidate["normal_world"][2]),
            "initial_separation_m": float(candidate["separation_m"]),
            "fixed_point_count": int(len(rows)),
            "negative_pose_count": int(candidate["negative_pose_count"]),
            "positive_pose_count": int(candidate["positive_pose_count"]),
        }
        for method in ("input_pgo", "side_aware", "unoriented"):
            values = global_distances[method][rows]
            stats = distance_stats(values, args.distance_cap)
            row[f"{method}_median_tls_m"] = stats["median_clipped_m"]
            row[f"{method}_p90_tls_m"] = stats["p90_clipped_m"]
            row[f"{method}_within_0_10m_rate"] = stats[
                "within_0_10m_rate"
            ]
            method_poses = aligned[method]
            negative_world = project_rows(
                negative_rows, local_points, pose_ids, method_poses
            )
            positive_world = project_rows(
                positive_rows, local_points, pose_ids, method_poses
            )
            negative_center, _ = projected_median_and_mad(
                negative_world, normal
            )
            positive_center, _ = projected_median_and_mad(
                positive_world, normal
            )
            row[f"{method}_surface_separation_m"] = abs(
                positive_center - negative_center
            )

        # Survey-map layer identity is frozen from the input-PGO projection.
        # Each mobile layer is associated independently, then summarized along
        # the candidate normal. This tests physical separation rather than a
        # generic nearest-point score diluted by floors, vegetation and clutter.
        input_negative = project_rows(
            negative_rows, local_points, pose_ids, aligned["input_pgo"]
        )
        input_positive = project_rows(
            positive_rows, local_points, pose_ids, aligned["input_pgo"]
        )
        negative_distance, negative_index = query_nearest(
            tls_tree, input_negative, args.tls_layer_match_cap
        )
        positive_distance, positive_index = query_nearest(
            tls_tree, input_positive, args.tls_layer_match_cap
        )
        negative_valid = np.isfinite(negative_distance)
        positive_valid = np.isfinite(positive_distance)
        row["negative_tls_match_rate"] = float(negative_valid.mean())
        row["positive_tls_match_rate"] = float(positive_valid.mean())
        row["negative_tls_unique_points"] = int(np.unique(
            negative_index[negative_valid]
        ).size)
        row["positive_tls_unique_points"] = int(np.unique(
            positive_index[positive_valid]
        ).size)
        tls_reference_separation = float("nan")
        negative_tls_mad = float("nan")
        positive_tls_mad = float("nan")
        if negative_valid.any() and positive_valid.any():
            negative_tls_center, negative_tls_mad = projected_median_and_mad(
                tls_points[negative_index[negative_valid]], normal
            )
            positive_tls_center, positive_tls_mad = projected_median_and_mad(
                tls_points[positive_index[positive_valid]], normal
            )
            tls_reference_separation = abs(
                positive_tls_center - negative_tls_center
            )
        row["tls_reference_separation_m"] = tls_reference_separation
        row["negative_tls_projection_mad_m"] = negative_tls_mad
        row["positive_tls_projection_mad_m"] = positive_tls_mad
        tls_validated = (
            row["negative_tls_match_rate"] >= args.tls_layer_min_match_rate
            and row["positive_tls_match_rate"] >= args.tls_layer_min_match_rate
            and row["negative_tls_unique_points"] >= args.tls_layer_min_points
            and row["positive_tls_unique_points"] >= args.tls_layer_min_points
            and np.isfinite(tls_reference_separation)
            and args.min_separation <= tls_reference_separation <= args.max_separation
            and negative_tls_mad <= args.tls_layer_max_mad
            and positive_tls_mad <= args.tls_layer_max_mad
        )
        row["tls_validated_double_surface"] = int(tls_validated)
        for method in ("input_pgo", "side_aware", "unoriented"):
            value = row[f"{method}_surface_separation_m"]
            row[f"{method}_tls_separation_error_m"] = (
                abs(value - tls_reference_separation)
                if np.isfinite(tls_reference_separation) else float("nan")
            )
        row["side_aware_minus_unoriented_median_m"] = (
            row["side_aware_median_tls_m"]
            - row["unoriented_median_tls_m"]
        )
        candidate_rows.append(row)
    candidate_csv = args.output_dir / "candidate_tls_source_data.csv"
    write_csv(candidate_csv, candidate_rows)
    del initial_world

    paired_delta = np.asarray([
        row["side_aware_minus_unoriented_median_m"]
        for row in candidate_rows
    ])
    tls_validated_rows = [
        row for row in candidate_rows if row["tls_validated_double_surface"]
    ]
    separation_delta = np.asarray([
        row["side_aware_tls_separation_error_m"]
        - row["unoriented_tls_separation_error_m"]
        for row in tls_validated_rows
    ])
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "protocol": {
            "point_identity": (
                "same downsampled local points and pose ids reprojected by every arm"
            ),
            "alignment": (
                "one SE3 Kabsch alignment from input PGO positions to interpolated "
                "truth; the identical transform is frozen for all BA arms"
            ),
            "tls_sampling": f"every {args.tls_stride}th raw TLS PCD row",
            "metric": "one-sided mobile-map to TLS nearest-neighbour distance",
            "outlier_policy": (
                f"all points retained and distances clipped at {args.distance_cap} m"
            ),
        },
        "parameters": {
            "root_voxel_size": params.root_voxel_size,
            "max_layer": params.max_layer,
            "downsample_leaf": params.downsample_leaf,
            "max_observation_range": params.max_range,
            "plane_thickness": params.plane_thickness,
            "min_separation": args.min_separation,
            "max_separation": args.max_separation,
            "max_abs_normal_z": args.max_abs_normal_z,
            "distance_cap": args.distance_cap,
            "tls_layer_match_cap": args.tls_layer_match_cap,
            "tls_layer_min_match_rate": args.tls_layer_min_match_rate,
            "tls_layer_min_points": args.tls_layer_min_points,
            "tls_layer_max_mad": args.tls_layer_max_mad,
        },
        "counts": {
            "poses": pose_count,
            "truth_matched_poses": int(truth_valid.sum()),
            "fixed_mobile_points": int(len(local_points)),
            "sampled_tls_points": int(len(tls_points)),
            "double_surface_diagnostics": len(diagnostics),
            "eligible_vertical_candidates": len(candidate_rows),
        },
        "global_map_accuracy": global_stats,
        "candidate_summary": {
            "candidate_count": len(candidate_rows),
            "side_aware_better_count": int(np.sum(paired_delta < 0.0)),
            "unoriented_better_count": int(np.sum(paired_delta > 0.0)),
            "ties": int(np.sum(paired_delta == 0.0)),
            "median_side_aware_minus_unoriented_tls_m": (
                float(np.median(paired_delta)) if len(paired_delta) else None
            ),
            "tls_validated_double_surface_count": len(tls_validated_rows),
            "tls_validated_side_aware_better_separation_count": int(
                np.sum(separation_delta < 0.0)
            ),
            "tls_validated_unoriented_better_separation_count": int(
                np.sum(separation_delta > 0.0)
            ),
            "tls_validated_median_side_aware_separation_error_m": (
                float(np.median([
                    row["side_aware_tls_separation_error_m"]
                    for row in tls_validated_rows
                ])) if tls_validated_rows else None
            ),
            "tls_validated_median_unoriented_separation_error_m": (
                float(np.median([
                    row["unoriented_tls_separation_error_m"]
                    for row in tls_validated_rows
                ])) if tls_validated_rows else None
            ),
            "tls_validated_median_error_delta_m": (
                float(np.median(separation_delta))
                if len(separation_delta) else None
            ),
        },
        "inputs": {
            "input_tum_sha256": sha256_file(args.input_tum),
            "side_aware_tum_sha256": sha256_file(args.side_aware_tum),
            "unoriented_tum_sha256": sha256_file(args.unoriented_tum),
            "truth_tum_sha256": sha256_file(args.truth_tum),
            "tls_map_sha256": sha256_file(args.tls_map),
            "frame_config_sha256": sha256_file(args.frame_config),
            "script_sha256": sha256_file(HERE),
        },
        "source_data": {
            "candidate_csv": candidate_csv.name,
            "candidate_csv_sha256": sha256_file(candidate_csv),
        },
    }
    (args.output_dir / "tls_map_accuracy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-tum", type=Path, required=True)
    parser.add_argument("--side-aware-tum", type=Path, required=True)
    parser.add_argument("--unoriented-tum", type=Path, required=True)
    parser.add_argument("--truth-tum", type=Path, required=True)
    parser.add_argument("--tls-map", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tls-stride", type=int, default=10)
    parser.add_argument("--root-voxel", type=float, default=4.0)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--downsample-leaf", type=float, default=0.4)
    parser.add_argument("--max-observation-range", type=float, default=30.0)
    parser.add_argument("--plane-thickness", type=float, default=0.12)
    parser.add_argument("--min-separation", type=float, default=0.04)
    parser.add_argument("--max-separation", type=float, default=0.50)
    parser.add_argument("--max-abs-normal-z", type=float, default=0.5)
    parser.add_argument("--distance-cap", type=float, default=1.0)
    parser.add_argument("--tls-layer-match-cap", type=float, default=0.25)
    parser.add_argument("--tls-layer-min-match-rate", type=float, default=0.50)
    parser.add_argument("--tls-layer-min-points", type=int, default=20)
    parser.add_argument("--tls-layer-max-mad", type=float, default=0.04)
    args = parser.parse_args()
    for name in (
        "input_tum", "side_aware_tum", "unoriented_tum", "truth_tum",
        "tls_map", "keyframe_dir", "frame_config",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    report = run(args)
    print(json.dumps({
        "counts": report["counts"],
        "global_map_accuracy": report["global_map_accuracy"],
        "candidate_summary": report["candidate_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
