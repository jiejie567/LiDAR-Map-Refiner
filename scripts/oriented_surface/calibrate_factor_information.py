#!/usr/bin/env python3
"""Calibrate oriented-factor information scale on frozen public sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import chi2

from rebuild_oriented_pose_graph import (
    OrientedFactorConfig,
    parse_g2o_edge,
    prepare_frame,
    remeasure,
    sha256_file,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_log_g2o(transform: np.ndarray) -> np.ndarray:
    """Return SE(3) logarithm in [translation, rotation] tangent order."""
    transform = np.asarray(transform, dtype=np.float64)
    omega = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-6:
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega_hat + omega_hat @ omega_hat / 12.0
        )
    else:
        coefficient = (
            1.0 / theta**2
            - (1.0 + math.cos(theta))
            / (2.0 * theta * math.sin(theta))
        )
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega_hat
            + coefficient * (omega_hat @ omega_hat)
        )
    translation = inverse_left_jacobian @ transform[:3, 3]
    return np.concatenate([translation, omega])


def tum_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    transforms = np.repeat(np.eye(4)[None], len(data), axis=0)
    transforms[:, :3, 3] = data[:, 1:4]
    transforms[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    return data[:, 0], transforms


def interpolated_truth(
    estimate_timestamps: np.ndarray, truth_path: Path, max_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    truth_t, truth_poses = tum_data(truth_path)
    right = np.searchsorted(truth_t, estimate_timestamps)
    right = np.clip(right, 1, len(truth_t) - 1)
    left = right - 1
    left_gap = estimate_timestamps - truth_t[left]
    right_gap = truth_t[right] - estimate_timestamps
    valid = (left_gap >= 0.0) & (right_gap >= 0.0) & (
        np.minimum(left_gap, right_gap) <= max_dt_s
    )
    interval = np.maximum(truth_t[right] - truth_t[left], 1e-12)
    fraction = np.clip(left_gap / interval, 0.0, 1.0)
    output = np.repeat(np.eye(4)[None], len(estimate_timestamps), axis=0)
    output[:, :3, 3] = (
        truth_poses[left, :3, 3] * (1.0 - fraction[:, None])
        + truth_poses[right, :3, 3] * fraction[:, None]
    )
    for index in np.nonzero(valid)[0]:
        rotations = Rotation.from_matrix(
            truth_poses[[left[index], right[index]], :3, :3]
        )
        output[index, :3, :3] = Slerp([0.0, 1.0], rotations)(
            [fraction[index]]
        ).as_matrix()[0]
    return output, valid


def truth_body_to_keyframe_from_manifest(
    manifest: dict, frame_config: dict,
) -> tuple[np.ndarray, dict]:
    dataset = manifest["dataset"]
    dataset_entry = frame_config["datasets"][dataset]
    contract = manifest.get("keyframe_export") or {}
    role = contract.get("keyframe_frame_role") or dataset_entry.get(
        "legacy_keyframe_frame_role"
    )
    if "keyframe_frames" in dataset_entry:
        if not role:
            raise ValueError(f"missing keyframe frame role for {dataset}")
        try:
            frame_entry = dataset_entry["keyframe_frames"][role]
        except KeyError as exc:
            raise KeyError(
                f"dataset {dataset} has no calibration for role {role}"
            ) from exc
    else:
        role = role or "legacy_unspecified"
        frame_entry = dataset_entry
    transform = np.asarray(
        frame_entry["T_truth_body_keyframe"], dtype=np.float64
    )
    if transform.shape != (4, 4):
        raise ValueError(f"invalid frame transform for {dataset}/{role}")
    return transform, {
        "keyframe_frame_role": role,
        "frame_contract_provenance": contract.get(
            "provenance", "frame_config_legacy_fallback"
        ),
        "frame_entry": frame_entry,
    }


def sequential_edges(path: Path) -> list[dict]:
    edges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        edge = parse_g2o_edge(line)
        if edge is not None and edge["sequential"]:
            edges.append(edge)
    return edges


def evenly_spaced(items: list[dict], maximum: int) -> list[dict]:
    if maximum <= 0 or len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=int)
    return [items[index] for index in np.unique(indices)]


def factor_error_g2o(
    measured_target_source: np.ndarray,
    truth_world_target: np.ndarray,
    truth_world_source: np.ndarray,
) -> np.ndarray:
    expected_target_source = (
        np.linalg.inv(truth_world_target) @ truth_world_source
    )
    # The estimator covariance is built for a left perturbation in target frame.
    left_gap = expected_target_source @ np.linalg.inv(measured_target_source)
    return se3_log_g2o(left_gap)


def collect_dataset(
    session: Path, config: OrientedFactorConfig, max_edges: int,
    max_truth_dt_s: float, frame_config: dict,
) -> tuple[list[dict], dict]:
    manifest = json.loads((session / "input_manifest.json").read_text())
    if manifest.get("truth_kind") != "se3":
        return [], {"dataset": manifest["dataset"], "skipped": "truth_not_se3"}
    timestamps, _ = tum_data(session / "optimized_poses_tum.txt")
    truth_poses, truth_valid = interpolated_truth(
        timestamps, Path(manifest["truth"]), max_truth_dt_s
    )
    truth_to_keyframe, frame_resolution = truth_body_to_keyframe_from_manifest(
        manifest, frame_config
    )
    truth_poses = truth_poses @ truth_to_keyframe
    edges = evenly_spaced(sequential_edges(session / "pose_graph.g2o"), max_edges)
    cache = {}
    rows = []
    for edge in edges:
        target_id = int(edge["target_id"])
        source_id = int(edge["source_id"])
        if not (truth_valid[target_id] and truth_valid[source_id]):
            continue
        result = remeasure(
            target_id, source_id, edge["transform"],
            session / "key_point_frame", config, cache,
        )
        if not result.valid:
            rows.append({
                "dataset": manifest["dataset"], "target_id": target_id,
                "source_id": source_id, "valid": False,
                "reason": result.reason,
            })
            continue
        error = factor_error_g2o(
            result.transform_target_source,
            truth_poses[target_id], truth_poses[source_id],
        )
        information = result.information_g2o
        raw_nis = float(error @ information @ error)
        rows.append({
            "dataset": manifest["dataset"],
            "target_id": target_id,
            "source_id": source_id,
            "valid": True,
            "reason": result.reason,
            "translation_error_m": float(np.linalg.norm(error[:3])),
            "rotation_error_deg": float(np.degrees(np.linalg.norm(error[3:]))),
            "raw_nis": raw_nis,
            "observable_rank": int(result.observable_rank),
            "overlap": float(result.overlap),
            "spatial_overlap": float(result.spatial_overlap),
            "rmse_m": float(result.inlier_rmse_m),
            "residual_scale_m": float(result.residual_scale_m),
            "information_upper_json": json.dumps([
                float(information[row, column])
                for row in range(6) for column in range(row, 6)
            ]),
            "error_g2o_json": json.dumps(error.tolist()),
        })
    return rows, {
        "dataset": manifest["dataset"],
        "requested_edges": len(edges),
        "truth_matched_edges": len(rows),
        "valid_factors": sum(bool(row["valid"]) for row in rows),
        "input_g2o_sha256": sha256_file(session / "pose_graph.g2o"),
        "input_tum_sha256": sha256_file(session / "optimized_poses_tum.txt"),
        "truth_sha256": sha256_file(Path(manifest["truth"])),
        "truth_to_keyframe": frame_resolution,
    }


def calibration_scales(
    rows: list[dict], calibration_datasets: set[str], coverage_quantile: float = 0.95,
) -> dict[str, float]:
    ratios = []
    for row in rows:
        if not row.get("valid") or row["dataset"] not in calibration_datasets:
            continue
        rank = max(1, int(row["observable_rank"]))
        raw_nis = float(row["raw_nis"])
        if np.isfinite(raw_nis) and raw_nis > 0.0:
            ratios.append(float(chi2.ppf(0.5, rank)) / raw_nis)
    valid_rows = [
        row for row in rows
        if row.get("valid") and row["dataset"] in calibration_datasets
        and np.isfinite(float(row["raw_nis"])) and float(row["raw_nis"]) > 0.0
    ]
    if not ratios or not valid_rows:
        raise RuntimeError("no valid calibration factors")
    median_scale = float(np.median(ratios))
    normalized_tail = [
        float(row["raw_nis"]) / chi2.ppf(
            0.95, max(1, int(row["observable_rank"]))
        )
        for row in valid_rows
    ]
    coverage_scale = float(
        1.0 / np.quantile(normalized_tail, coverage_quantile)
    )
    return {
        "median_chi_square_scale": median_scale,
        "coverage_scale": coverage_scale,
        "selected_conservative_scale": min(median_scale, coverage_scale),
        "coverage_quantile": float(coverage_quantile),
    }


def calibration_scale(rows: list[dict], calibration_datasets: set[str]) -> float:
    return calibration_scales(rows, calibration_datasets)[
        "selected_conservative_scale"
    ]


def summarize(rows: list[dict], scale: float, calibration: set[str]) -> dict:
    summary = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        valid = [row for row in rows if row["dataset"] == dataset and row.get("valid")]
        if not valid:
            summary[dataset] = {"valid_factors": 0}
            continue
        calibrated_nis = np.asarray([scale * float(row["raw_nis"]) for row in valid])
        thresholds = np.asarray([
            chi2.ppf(0.95, max(1, int(row["observable_rank"]))) for row in valid
        ])
        summary[dataset] = {
            "split": "calibration" if dataset in calibration else "held_out",
            "valid_factors": len(valid),
            "median_translation_error_m": float(np.median([
                float(row["translation_error_m"]) for row in valid
            ])),
            "median_rotation_error_deg": float(np.median([
                float(row["rotation_error_deg"]) for row in valid
            ])),
            "median_raw_nis": float(np.median([
                float(row["raw_nis"]) for row in valid
            ])),
            "median_calibrated_nis": float(np.median(calibrated_nis)),
            "calibrated_95pct_coverage": float(np.mean(calibrated_nis <= thresholds)),
            "median_oriented_overlap": float(np.median([
                float(row["overlap"]) for row in valid
            ])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="building_day,kth,ntu,spires")
    parser.add_argument("--calibration-datasets", default="building_day,spires")
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument("--max-edges-per-dataset", type=int, default=50)
    parser.add_argument("--max-truth-dt", type=float, default=0.05)
    parser.add_argument("--voxel", type=float, default=0.20)
    parser.add_argument("--normal-radius", type=float, default=0.60)
    parser.add_argument("--max-correspondence", type=float, default=0.60)
    parser.add_argument("--normal-gate-deg", type=float, default=45.0)
    parser.add_argument("--min-overlap", type=float, default=0.20)
    parser.add_argument("--min-correspondences", type=int, default=80)
    parser.add_argument("--translation-sigma-floor-m", type=float, default=0.0)
    parser.add_argument("--rotation-sigma-floor-deg", type=float, default=0.0)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    config = OrientedFactorConfig(
        voxel_size_m=args.voxel, normal_radius_m=args.normal_radius,
        max_correspondence_m=args.max_correspondence,
        normal_gate_deg=args.normal_gate_deg, min_overlap=args.min_overlap,
        min_correspondences=args.min_correspondences,
        translation_sigma_floor_m=args.translation_sigma_floor_m,
        rotation_sigma_floor_rad=math.radians(args.rotation_sigma_floor_deg),
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    calibration = {
        item.strip() for item in args.calibration_datasets.split(",") if item.strip()
    }
    frame_config_path = args.frame_config.expanduser().resolve()
    frame_config = json.loads(frame_config_path.read_text(encoding="utf-8"))
    rows = []
    inventories = []
    for dataset in datasets:
        dataset_rows, inventory = collect_dataset(
            args.frozen_root / dataset, config,
            args.max_edges_per_dataset, args.max_truth_dt, frame_config,
        )
        rows.extend(dataset_rows)
        inventories.append(inventory)
        print(json.dumps(inventory, sort_keys=True))
    scales = calibration_scales(rows, calibration)
    scale = scales["selected_conservative_scale"]
    for row in rows:
        row["calibrated_nis"] = (
            scale * float(row["raw_nis"]) if row.get("valid") else None
        )
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "factor_calibration_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "conservative_chi_square_scale_on_preregistered_sequences",
        "calibration_datasets": sorted(calibration),
        "held_out_datasets": sorted(set(datasets) - calibration),
        "information_scale": scale,
        "calibration_scales": scales,
        "config_before_scale": asdict(config),
        "frame_config": str(frame_config_path),
        "frame_config_sha256": sha256_file(frame_config_path),
        "inventories": inventories,
        "summary": summarize(rows, scale, calibration),
    }
    (output_dir / "factor_calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"information_scale": scale, "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
