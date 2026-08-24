#!/usr/bin/env python3
"""Verify a rebuild and emit the C1 gate report plus source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from proposal_ledger import ProposalLedger


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
SE3_EVAL = REPO / "experiments/mcd_new_sequences/evaluate_trajectory.py"
POSITION_EVAL = HERE / "ate_eval.py"
POSITION_SEQUENCE = {"hall02": "hall_02", "hall04": "hall_04"}
FRAME_CONFIG = (
    REPO / "tools/manual_loop_closure/config/oriented_surface_dataset_frames.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quat_matrix(values) -> np.ndarray:
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("invalid zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tum_transforms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, dtype=float))
    transforms = np.repeat(np.eye(4)[None], len(data), axis=0)
    transforms[:, :3, 3] = data[:, 1:4]
    for index, quaternion in enumerate(data[:, 4:8]):
        transforms[index, :3, :3] = quat_matrix(quaternion)
    return data[:, 0], transforms


def target_to_source_truth_error(
    measured_target_source: np.ndarray,
    truth_world_source: np.ndarray,
    truth_world_target: np.ndarray,
) -> tuple[float, float]:
    """Return SE(3) error for the optimizer's target-to-source convention."""
    expected = np.linalg.inv(truth_world_target) @ truth_world_source
    gap = np.linalg.inv(measured_target_source) @ expected
    translation_m = float(np.linalg.norm(gap[:3, 3]))
    cosine = np.clip((np.trace(gap[:3, :3]) - 1) / 2, -1, 1)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_deg


def nearest_truth(
    estimate_timestamps: np.ndarray,
    truth_path: Path,
    truth_kind: str,
    max_dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.atleast_2d(np.loadtxt(truth_path, dtype=float))
    truth_t = truth[:, 0]
    indices = np.searchsorted(truth_t, estimate_timestamps)
    indices = np.clip(indices, 1, len(truth_t) - 1)
    use_left = (
        np.abs(truth_t[indices - 1] - estimate_timestamps)
        < np.abs(truth_t[indices] - estimate_timestamps)
    )
    indices[use_left] -= 1
    valid = np.abs(truth_t[indices] - estimate_timestamps) <= max_dt
    transforms = np.repeat(np.eye(4)[None], len(estimate_timestamps), axis=0)
    transforms[:, :3, 3] = truth[indices, 1:4]
    orientation_valid = np.zeros(len(estimate_timestamps), dtype=bool)
    if truth_kind == "se3" and truth.shape[1] >= 8:
        for index, quaternion in enumerate(truth[indices, 4:8]):
            try:
                transforms[index, :3, :3] = quat_matrix(quaternion)
                orientation_valid[index] = True
            except ValueError:
                pass
    return transforms, valid, orientation_valid


def interpolated_truth_in_keyframe_frame(
    estimate_timestamps: np.ndarray,
    truth_path: Path,
    dataset: str,
    frame_config: dict,
    max_dt: float = 0.05,
    keyframe_frame_role: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate SE(3) truth and express it at the keyframe sensor origin.

    Public datasets publish truth at different body frames.  Relative loop
    error is invariant to a *world* alignment but not to changing the body
    origin, so the official body-to-keyframe lever arm must be applied before
    labelling a factor.  Nearest-neighbour truth also biases fast rotations;
    translation is therefore linearly interpolated and rotation uses SLERP.
    """
    truth_t, truth_poses = tum_transforms(truth_path)
    right = np.searchsorted(truth_t, estimate_timestamps)
    right = np.clip(right, 1, len(truth_t) - 1)
    left = right - 1
    left_gap = estimate_timestamps - truth_t[left]
    right_gap = truth_t[right] - estimate_timestamps
    valid = (
        (left_gap >= 0.0)
        & (right_gap >= 0.0)
        & (np.minimum(left_gap, right_gap) <= max_dt)
    )
    interval = np.maximum(truth_t[right] - truth_t[left], 1e-12)
    fraction = np.clip(left_gap / interval, 0.0, 1.0)
    output = np.repeat(np.eye(4)[None], len(estimate_timestamps), axis=0)
    output[:, :3, 3] = (
        truth_poses[left, :3, 3] * (1.0 - fraction[:, None])
        + truth_poses[right, :3, 3] * fraction[:, None]
    )
    for index in np.nonzero(valid)[0]:
        pair = Rotation.from_matrix(
            truth_poses[[left[index], right[index]], :3, :3]
        )
        output[index, :3, :3] = Slerp([0.0, 1.0], pair)(
            [fraction[index]]
        ).as_matrix()[0]
    truth_body_to_keyframe, _ = resolve_truth_body_to_keyframe(
        dataset, frame_config, keyframe_frame_role
    )
    if truth_body_to_keyframe.shape != (4, 4):
        raise ValueError(f"invalid truth-frame transform for {dataset}")
    output = output @ truth_body_to_keyframe
    return output, valid, np.asarray(valid, dtype=bool)


def resolve_truth_body_to_keyframe(
    dataset: str,
    frame_config: dict,
    keyframe_frame_role: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Resolve a dataset transform using the frame frozen with the session.

    Schema-v1 configs are retained only for small unit fixtures. Production
    schema-v2 configs carry separate transforms for ``lidar_imu`` and
    ``base_link`` so regenerating a session cannot silently change truth labels.
    """
    try:
        dataset_entry = frame_config["datasets"][dataset]
    except KeyError as exc:
        raise KeyError(f"missing truth-frame transform for {dataset}") from exc
    if "keyframe_frames" in dataset_entry:
        role = keyframe_frame_role or dataset_entry.get(
            "legacy_keyframe_frame_role"
        )
        if not role:
            raise ValueError(
                f"keyframe frame role is required for dataset {dataset}"
            )
        try:
            frame_entry = dataset_entry["keyframe_frames"][role]
        except KeyError as exc:
            raise KeyError(
                f"dataset {dataset} has no calibration for keyframe role {role}"
            ) from exc
    else:
        role = keyframe_frame_role or "legacy_unspecified"
        frame_entry = dataset_entry
    transform = np.asarray(frame_entry["T_truth_body_keyframe"], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"invalid truth-frame transform for {dataset}/{role}")
    return transform, {
        "keyframe_frame_role": role,
        "keyframe_frame_name": frame_entry.get("name"),
        "truth_frame": dataset_entry.get("truth_frame"),
        "calibration_source": frame_entry.get("source"),
    }


def manifest_keyframe_frame_role(manifest: dict, frame_config: dict) -> tuple[str, str]:
    contract = manifest.get("keyframe_export")
    if contract:
        if contract.get("point_frame_equals_pose_frame") is not True:
            raise ValueError("keyframe points and poses must share one local frame")
        role = contract.get("keyframe_frame_role")
        if not role:
            raise ValueError("keyframe_export is missing keyframe_frame_role")
        return str(role), str(contract.get("provenance", "manifest"))
    dataset_entry = frame_config["datasets"][manifest["dataset"]]
    role = dataset_entry.get("legacy_keyframe_frame_role")
    if role is None:
        # Backward compatibility for schema-v1 test fixtures only.
        role = "legacy_unspecified"
    return str(role), "frame_config_legacy_fallback"


def trajectory_metric(tag: str, tum_path: Path, manifest: dict) -> float | None:
    truth_value = manifest.get("truth")
    if not truth_value or not tum_path.is_file():
        return None
    truth = Path(truth_value)
    if manifest.get("truth_kind") == "se3":
        completed = subprocess.run(
            [sys.executable, str(SE3_EVAL), "--estimate", str(tum_path),
             "--ground-truth", str(truth)],
            capture_output=True, text=True,
        )
        try:
            return float(json.loads(completed.stdout)["position_error_m"]["rmse"])
        except Exception:
            return None
    completed = subprocess.run(
        [sys.executable, str(POSITION_EVAL), str(tum_path), str(truth),
         POSITION_SEQUENCE[tag]],
        capture_output=True, text=True,
    )
    try:
        return float(completed.stdout.split("=")[1].split("cm")[0]) / 100.0
    except Exception:
        return None


def verify_inputs(session: Path, manifest: dict) -> None:
    for name, expected in manifest["input_sha256"].items():
        actual = sha256_file(session / name)
        if actual != expected:
            raise ValueError(f"input hash mismatch: {session / name}")


def infer_stage0_output(session: Path, summary: dict) -> Path | None:
    explicit = summary.get("stage0_audited_output_dir")
    if explicit:
        return Path(explicit)
    candidates = sorted(
        path for path in (session / "manual_loop_runs").glob("*_auto_r1")
        if (path / "optimized_poses_tum.txt").is_file()
    )
    return candidates[0] if candidates else None


def trial_factor_measurements(session: Path) -> dict[str, dict]:
    """Recover final factor measurements from immutable probation CSVs.

    Schema-v1 ledgers recorded the factor information but, before the schema-v2
    fix, omitted the remeasured transform.  Trial directories are content
    artifacts named by proposal id, so this reconstruction is deterministic
    and does not alter the append-only ledger.
    """
    output = {}
    patterns = (
        "*_stage0_trials/*/manual_loop_constraints.csv",
        "*_auto_r*/proposal_trials/*/manual_loop_constraints.csv",
    )
    for pattern in patterns:
        for path in (session / "manual_loop_runs").glob(pattern):
            proposal = path.parent.name
            try:
                rows = list(csv.DictReader(path.open(encoding="utf-8")))
            except Exception:
                continue
            if not rows:
                continue
            row = rows[-1]
            try:
                output[proposal] = {
                    "translation_m": [
                        float(row[key]) for key in ("tx", "ty", "tz")
                    ],
                    "quaternion_xyzw": [
                        float(row[key]) for key in ("qx", "qy", "qz", "qw")
                    ],
                    "source": str(path.resolve()),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return output


def load_gravity_vectors(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        return {}
    output = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                output[int(row["index"])] = np.asarray([
                    float(row["up_x"]), float(row["up_y"]),
                    float(row["up_z"]),
                ])
            except (KeyError, TypeError, ValueError):
                continue
    return output


def gravity_error_deg(
    measured_target_source: np.ndarray,
    source_up: np.ndarray | None,
    target_up: np.ndarray | None,
) -> float | None:
    if source_up is None or target_up is None:
        return None
    source = np.asarray(source_up, dtype=float)
    target = np.asarray(target_up, dtype=float)
    if (
        not np.all(np.isfinite(source)) or not np.all(np.isfinite(target))
        or np.linalg.norm(source) < 1e-9 or np.linalg.norm(target) < 1e-9
    ):
        return None
    predicted = measured_target_source[:3, :3] @ (
        source / np.linalg.norm(source)
    )
    cosine = np.clip(
        np.dot(predicted, target / np.linalg.norm(target)), -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cosine)))


def proposal_truth_rows(
    tag: str,
    session: Path,
    manifest: dict,
    summary: dict,
    records: list[dict],
    frame_config: dict,
) -> list[dict]:
    if not manifest.get("truth"):
        return []
    estimate_t, _ = tum_transforms(session / "optimized_poses_tum.txt")
    if manifest["truth_kind"] == "se3":
        frame_role, frame_provenance = manifest_keyframe_frame_role(
            manifest, frame_config
        )
        truth, valid, orientation_valid = interpolated_truth_in_keyframe_frame(
            estimate_t,
            Path(manifest["truth"]),
            manifest["dataset"],
            frame_config,
            keyframe_frame_role=frame_role,
        )
    else:
        frame_role = "position_truth_not_applicable"
        frame_provenance = "not_applicable"
        truth, valid, orientation_valid = nearest_truth(
            estimate_t, Path(manifest["truth"]), manifest["truth_kind"]
        )
    final_active = {
        item["original_proposal_id"]
        for item in records
        if item.get("source") == "proposal_finalization"
        and item.get("decision") == "retained_final"
    }
    recovered_factor_measurements = trial_factor_measurements(session)
    gravity_vectors = load_gravity_vectors(
        session / "scan_context_gravity.csv"
    )
    output = []
    for item in records:
        if item.get("source") not in {"stage0_descriptor", "map_inconsistency"}:
            continue
        if not (item.get("gicp") or {}).get("transform_target_source"):
            continue
        source_id = int(item["source_id"])
        target_id = int(item["target_id"])
        gicp_transform_data = item["gicp"]["transform_target_source"]
        factor_transform_data = (
            (item.get("loop_factor") or {}).get("transform_target_source")
            or recovered_factor_measurements.get(item["proposal_id"])
        )
        transform_data = factor_transform_data or gicp_transform_data
        row = {
            "dataset": tag,
            "proposal_id": item["proposal_id"],
            "source": item["source"],
            "keyframe_frame_role": frame_role,
            "frame_contract_provenance": frame_provenance,
            "round": item["round"],
            "source_id": source_id,
            "target_id": target_id,
            "decision": item["decision"],
            "reason": item.get("reason"),
            "active_final": item["proposal_id"] in final_active,
            "descriptor_distance": item.get("descriptor_distance"),
            "gicp_fitness": item["gicp"].get("fitness"),
            "gicp_rmse_m": item["gicp"].get("inlier_rmse_m"),
            "gicp_gate_passed": (
                item["gicp"].get("fitness", 0.0)
                >= summary["gates"]["min_overlap"]
                and item["gicp"].get("inlier_rmse_m", float("inf"))
                <= summary["gates"]["max_rmse_m"]
            ),
            "gravity_available": False,
            "gravity_error_deg": None,
            "gravity_gate_passed_1deg": None,
            "odometry_budget_occupancy": (item.get("odometry_budget") or {}).get(
                "budget_occupancy"),
            "factor_residual_m": item.get("factor_residual_m"),
            "factor_nis": item.get("factor_nis"),
            "factor_consistency_score": item.get(
                "factor_consistency_score"),
            "measurement_source": (
                "oriented_factor" if factor_transform_data else "gicp"
            ),
            "measurement_tx": transform_data["translation_m"][0],
            "measurement_ty": transform_data["translation_m"][1],
            "measurement_tz": transform_data["translation_m"][2],
            "measurement_qx": transform_data["quaternion_xyzw"][0],
            "measurement_qy": transform_data["quaternion_xyzw"][1],
            "measurement_qz": transform_data["quaternion_xyzw"][2],
            "measurement_qw": transform_data["quaternion_xyzw"][3],
            "gt_available": False,
            "gt_endpoint_distance_m": None,
            "relative_translation_error_m": None,
            "relative_rotation_error_deg": None,
            "gicp_relative_translation_error_m": None,
            "gicp_relative_rotation_error_deg": None,
            "factor_relative_translation_error_m": None,
            "factor_relative_rotation_error_deg": None,
            "before_separation_m": (item.get("fixed_evidence") or {}).get(
                "before_separation_m"),
            "after_separation_m": (item.get("fixed_evidence") or {}).get(
                "after_separation_m"),
        }
        if (
            source_id < len(valid) and target_id < len(valid)
            and valid[source_id] and valid[target_id]
        ):
            row["gt_available"] = True
            row["gt_endpoint_distance_m"] = float(np.linalg.norm(
                truth[source_id, :3, 3] - truth[target_id, :3, 3]
            ))
            for prefix, candidate_transform in (
                ("gicp", gicp_transform_data),
                ("factor", factor_transform_data),
            ):
                if not candidate_transform:
                    continue
                measured = np.eye(4)
                measured[:3, 3] = candidate_transform["translation_m"]
                measured[:3, :3] = quat_matrix(
                    candidate_transform["quaternion_xyzw"])
                translation_error, rotation_error = target_to_source_truth_error(
                    measured, truth[source_id], truth[target_id]
                )
                row[f"{prefix}_relative_translation_error_m"] = translation_error
                if orientation_valid[source_id] and orientation_valid[target_id]:
                    row[f"{prefix}_relative_rotation_error_deg"] = rotation_error
            effective_prefix = "factor" if factor_transform_data else "gicp"
            row["relative_translation_error_m"] = row[
                f"{effective_prefix}_relative_translation_error_m"
            ]
            row["relative_rotation_error_deg"] = row[
                f"{effective_prefix}_relative_rotation_error_deg"
            ]
        gicp_matrix = np.eye(4)
        gicp_matrix[:3, 3] = gicp_transform_data["translation_m"]
        gicp_matrix[:3, :3] = quat_matrix(
            gicp_transform_data["quaternion_xyzw"]
        )
        gravity_error = gravity_error_deg(
            gicp_matrix,
            gravity_vectors.get(source_id),
            gravity_vectors.get(target_id),
        )
        if gravity_error is not None:
            row["gravity_available"] = True
            row["gravity_error_deg"] = gravity_error
            row["gravity_gate_passed_1deg"] = gravity_error <= 1.0
        output.append(row)
    return output


def evaluate(root: Path, frame_config_path: Path = FRAME_CONFIG) -> dict:
    frame_config_path = frame_config_path.expanduser().resolve()
    frame_config = json.loads(frame_config_path.read_text(encoding="utf-8"))
    dataset_reports = {}
    proposal_rows = []
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = session / "input_manifest.json"
        summary_path = session / "auto_repair_summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            continue
        tag = session.name
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        verify_inputs(session, manifest)
        ledger_path = Path(summary["proposal_ledger"]["path"])
        records = ProposalLedger.verify(ledger_path)
        if sha256_file(ledger_path) != summary["proposal_ledger_sha256"]:
            raise ValueError(f"ledger hash mismatch: {tag}")
        rows = proposal_truth_rows(
            tag, session, manifest, summary, records, frame_config
        )
        proposal_rows.extend(rows)
        terminal = Path(summary["terminal_output_dir"])
        stage0 = infer_stage0_output(session, summary)
        odometry_ate = trajectory_metric(
            tag, session / "optimized_poses_tum.txt", manifest)
        stage0_ate = trajectory_metric(
            tag, stage0 / "optimized_poses_tum.txt", manifest
        ) if stage0 else None
        terminal_ate = trajectory_metric(
            tag, terminal / "optimized_poses_tum.txt", manifest)
        active_map = [
            row for row in rows
            if row["source"] == "map_inconsistency" and row["active_final"]
        ]
        improvement = None
        if stage0_ate and terminal_ate is not None:
            improvement = (stage0_ate - terminal_ate) / stage0_ate
        dataset_reports[tag] = {
            "odometry_ate_m": odometry_ate,
            "stage0_audited_ate_m": stage0_ate,
            "terminal_ate_m": terminal_ate,
            "stage0_to_terminal_relative_improvement": improvement,
            "active_constraints": summary["active_constraint_count"],
            "active_map_recovery_constraints": len(active_map),
            "rounds": summary["rounds_completed"],
            "keyframe_frame_role": (
                manifest.get("keyframe_export", {}).get("keyframe_frame_role")
                or frame_config["datasets"].get(
                    manifest["dataset"], {}
                ).get("legacy_keyframe_frame_role")
            ),
            "frame_contract_provenance": (
                manifest.get("keyframe_export", {}).get("provenance")
                or "frame_config_legacy_fallback"
            ),
            "input_manifest_sha256": sha256_file(manifest_path),
            "proposal_ledger_sha256": summary["proposal_ledger_sha256"],
        }

    active_map_rows = [
        row for row in proposal_rows
        if row["source"] == "map_inconsistency" and row["active_final"]
    ]
    full_truth_rows = [
        row for row in active_map_rows
        if row["relative_translation_error_m"] is not None
        and row["relative_rotation_error_deg"] is not None
    ]
    recovering_datasets = {
        row["dataset"] for row in active_map_rows if row["gt_available"]
    }
    improved_datasets = {
        tag for tag, item in dataset_reports.items()
        if item["stage0_to_terminal_relative_improvement"] is not None
        and item["stage0_to_terminal_relative_improvement"] >= 0.10
    }
    all_truth_safe = bool(active_map_rows) and len(full_truth_rows) == len(active_map_rows)
    if all_truth_safe:
        all_truth_safe = all(
            row["relative_translation_error_m"] <= 1.0
            and row["relative_rotation_error_deg"] <= 5.0
            for row in full_truth_rows
        )
    control_reports = {
        tag: item for tag, item in dataset_reports.items()
        if item["odometry_ate_m"] is not None and item["odometry_ate_m"] <= 0.25
    }
    controls_safe = bool(control_reports) and all(
        item["terminal_ate_m"] is not None
        and item["terminal_ate_m"] - item["odometry_ate_m"]
        <= max(0.02, 0.02 * item["odometry_ate_m"])
        for item in control_reports.values()
    )
    gates = {
        "at_least_two_public_gt_datasets_with_recovery": len(recovering_datasets) >= 2,
        "at_least_five_retained_map_recovery_loops": len(active_map_rows) >= 5,
        "all_retained_loops_within_1m_5deg": all_truth_safe,
        "at_least_two_sequences_improve_over_stage0_by_10pct": len(improved_datasets) >= 2,
        "accurate_controls_within_2cm_or_2pct": controls_safe,
    }
    return {
        "schema_version": 2,
        "loop_truth_protocol": {
            "measurement_convention": "target_to_source",
            "expected_transform": "inverse(T_world_target) @ T_world_source",
            "orientation_metric": "SO(3) geodesic angle",
            "timestamp_association": (
                "linear translation interpolation and SO(3) SLERP; nearest "
                "truth endpoint must be within 0.05 s"
            ),
            "frame_policy": (
                "world_T_keyframe = world_T_truth_body @ "
                "T_truth_body_keyframe using audited public calibration"
            ),
            "frame_config": str(frame_config_path),
            "frame_config_sha256": sha256_file(frame_config_path),
        },
        "dataset_reports": dataset_reports,
        "proposal_rows": proposal_rows,
        "c1": {
            "recovering_datasets": sorted(recovering_datasets),
            "retained_map_recovery_loop_count": len(active_map_rows),
            "improved_datasets": sorted(improved_datasets),
            "gates": gates,
            "passed": all(gates.values()),
            "recommended_mainline": (
                "GhostLoop: auditing and recovering"
                if all(gates.values())
                else "LoopAudit: recovery claim demoted; evaluate C2"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("--frame-config", type=Path, default=FRAME_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Write derived evidence outside the immutable rebuild root",
    )
    args = parser.parse_args()
    root = args.rebuild_root.expanduser().resolve()
    report = evaluate(root, args.frame_config)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "c1_source_data.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "proposal_source_data.csv"
    fieldnames = [
        "dataset", "proposal_id", "source", "keyframe_frame_role",
        "frame_contract_provenance", "round", "source_id", "target_id",
        "decision", "reason", "active_final", "descriptor_distance",
        "gicp_fitness", "gicp_rmse_m", "gicp_gate_passed",
        "gravity_available", "gravity_error_deg",
        "gravity_gate_passed_1deg",
        "odometry_budget_occupancy", "factor_residual_m",
        "factor_nis", "factor_consistency_score", "measurement_source",
        "measurement_tx", "measurement_ty", "measurement_tz",
        "measurement_qx", "measurement_qy", "measurement_qz", "measurement_qw",
        "gt_available", "gt_endpoint_distance_m",
        "relative_translation_error_m", "relative_rotation_error_deg",
        "gicp_relative_translation_error_m",
        "gicp_relative_rotation_error_deg",
        "factor_relative_translation_error_m",
        "factor_relative_rotation_error_deg",
        "before_separation_m", "after_separation_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["proposal_rows"])
    evidence = {
        json_path.name: sha256_file(json_path),
        csv_path.name: sha256_file(csv_path),
    }
    (output_dir / "c1_evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["c1"], indent=2))


if __name__ == "__main__":
    main()
