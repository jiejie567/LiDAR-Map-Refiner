#!/usr/bin/env python3
"""Frozen-pair A/B for TRO M3 observation-oriented loop refinement.

The experiment has a strict two-phase boundary.  Measurement and PGO never
open ground truth.  Only after both plain and M3 trajectories exist does the
evaluation phase load truth and attach loop-/trajectory-level errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GUI = REPO / "tools/manual_loop_closure/gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from manual_loop_closure.registration import (  # noqa: E402
    RegistrationConfig,
    RegistrationWorkspace,
)
from manual_loop_closure.trajectory_io import load_tum_trajectory  # noqa: E402
from evaluate_icra2027_rebuild import (  # noqa: E402
    FRAME_CONFIG,
    interpolated_truth_in_keyframe_frame,
    manifest_keyframe_frame_role,
    nearest_truth,
    target_to_source_truth_error,
    trajectory_metric,
)


OPTIMIZER_CLI = GUI / "manual_loop_closure/python_optimizer/cli.py"
DEFAULT_DATASETS = (
    "building_day", "kth", "ntu", "spires", "hall02", "hall04"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_from_row(row: dict[str, str]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [float(row[key]) for key in ("tx", "ty", "tz")]
    transform[:3, :3] = Rotation.from_quat(
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
    ).as_matrix()
    return transform


def _put_matrix(row: dict[str, str], transform: np.ndarray) -> None:
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    for key, value in zip(("tx", "ty", "tz"), transform[:3, 3]):
        row[key] = f"{float(value):.12f}"
    for key, value in zip(("qx", "qy", "qz", "qw"), quaternion):
        row[key] = f"{float(value):.12f}"


def _se3_gap(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    gap = np.linalg.inv(reference) @ candidate
    translation = float(np.linalg.norm(gap[:3, 3]))
    cosine = float(np.clip((np.trace(gap[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(np.degrees(np.arccos(cosine)))


def _read_constraints(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_constraints(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plain_rows_from_gicp_ledger(
    session: Path,
    frozen_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Replace only measurements with the matching immutable raw GICP rows."""
    summary = json.loads(
        (session / "auto_repair_summary.json").read_text(encoding="utf-8")
    )
    ledger_path = Path(summary["proposal_ledger"]["path"])
    queues: dict[tuple[int, int], list[np.ndarray]] = {}
    with ledger_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") not in {
                "stage0_descriptor", "map_inconsistency"
            }:
                continue
            payload = (record.get("gicp") or {}).get("transform_target_source")
            if not payload:
                continue
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = payload["translation_m"]
            transform[:3, :3] = Rotation.from_quat(
                payload["quaternion_xyzw"]
            ).as_matrix()
            pair = (int(record["source_id"]), int(record["target_id"]))
            queues.setdefault(pair, []).append(transform)

    output: list[dict[str, str]] = []
    for frozen in frozen_rows:
        pair = (int(frozen["source_id"]), int(frozen["target_id"]))
        candidates = queues.get(pair)
        if not candidates:
            raise ValueError(f"No raw GICP ledger measurement for pair {pair}")
        row = dict(frozen)
        _put_matrix(row, candidates.pop(0))
        output.append(row)
    return output


def _run_optimizer(
    *,
    session: Path,
    constraints: Path,
    output: Path,
    resume: bool,
) -> dict:
    report_path = output / "manual_loop_report.json"
    trajectory_path = output / "optimized_poses_tum.txt"
    if resume and report_path.is_file() and trajectory_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite optimizer output: {output}")
    output.mkdir(parents=True)
    command = [
        sys.executable,
        "-u",
        str(OPTIMIZER_CLI),
        "--session-root", str(session),
        "--g2o", str(session / "pose_graph.g2o"),
        "--tum", str(session / "optimized_poses_tum.txt"),
        "--keyframe-dir", str(session / "key_point_frame"),
        "--constraints-csv", str(constraints),
        "--output-dir", str(output),
        "--optimize-mode", "gnc_tls",
        "--manual-information-scale", "0.1",
        "--loop-correlation-window-keyframes", "0",
        "--loop-cluster-information-budget", "0",
        "--loop-correlation-policy", "pair",
        "--loop-cluster-allocation", "equal",
        "--skip-map-build",
        "--skip-graph-plot",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (output / "solver.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Optimizer failed for {output}; inspect solver.log")
    return json.loads(report_path.read_text(encoding="utf-8"))


def measurement_and_pgo_phase(
    *,
    dataset: str,
    session: Path,
    frozen_csv: Path,
    output: Path,
    resume: bool,
    plain_measurement_source: str,
) -> dict:
    """Generate both arms without reading any truth artifact."""
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "measurement_results_no_truth.json"
    m3_csv = output / "m3_constraints.csv"
    plain_csv = output / "plain_constraints.csv"
    if resume and result_path.is_file() and m3_csv.is_file() and plain_csv.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        fieldnames, frozen_rows = _read_constraints(frozen_csv)
        if plain_measurement_source == "gicp_ledger":
            plain_rows = _plain_rows_from_gicp_ledger(session, frozen_rows)
        elif plain_measurement_source == "frozen_csv":
            plain_rows = [dict(row) for row in frozen_rows]
        else:
            raise ValueError(
                f"Unsupported plain measurement source: {plain_measurement_source}"
            )
        _write_constraints(plain_csv, fieldnames, plain_rows)
        trajectory = load_tum_trajectory(session / "optimized_poses_tum.txt")
        workspace = RegistrationWorkspace(session / "key_point_frame", trajectory)
        summary = json.loads(
            (session / "auto_repair_summary.json").read_text(encoding="utf-8")
        )
        near = dict(summary["registration"]["near"])
        near.update({
            "normal_gate": True,
            "normal_gate_max_angle_deg": 60.0,
            "normal_radius": 0.25,
            "max_iterations": 30,
        })
        config = RegistrationConfig(**near)
        m3_rows: list[dict[str, str]] = []
        measurements: list[dict] = []
        print(f"[{dataset}] M3 refining {len(plain_rows)} frozen pairs", flush=True)
        for index, plain_row in enumerate(plain_rows, start=1):
            row = dict(plain_row)
            source_id = int(row["source_id"])
            target_id = int(row["target_id"])
            plain_target_source = _matrix_from_row(row)
            desired_world_source = (
                trajectory.transforms_world_sensor[target_id]
                @ plain_target_source
            )
            delta_local = (
                np.linalg.inv(trajectory.transforms_world_sensor[source_id])
                @ desired_world_source
            )
            try:
                registration = workspace.run_gicp(
                    source_id=source_id,
                    target_id=target_id,
                    delta_transform_local=delta_local,
                    config=config,
                )
                diagnostics = registration.oriented_diagnostics
                if diagnostics is None:
                    raise RuntimeError("M3 returned no oriented diagnostics")
                m3_target_source = registration.transform_target_source_final
                use_m3 = bool(diagnostics.valid)
                reason = (
                    "oriented_refinement_valid"
                    if use_m3
                    else f"fallback_plain:{diagnostics.termination_reason}"
                )
            except Exception as exc:
                diagnostics = None
                m3_target_source = plain_target_source
                use_m3 = False
                reason = f"fallback_plain:exception:{type(exc).__name__}:{exc}"
            selected = m3_target_source if use_m3 else plain_target_source
            _put_matrix(row, selected)
            m3_rows.append(row)
            shift_t, shift_r = _se3_gap(plain_target_source, selected)
            measurements.append({
                "row_index": index - 1,
                "source_id": source_id,
                "target_id": target_id,
                "used_m3": use_m3,
                "decision_reason": reason,
                "plain_transform_target_source": plain_target_source.tolist(),
                "selected_transform_target_source": selected.tolist(),
                "measurement_shift_translation_m": shift_t,
                "measurement_shift_rotation_deg": shift_r,
                "diagnostics": asdict(diagnostics) if diagnostics else None,
            })
            print(
                f"[{dataset}] {index}/{len(plain_rows)} "
                f"{source_id}->{target_id} {reason} "
                f"shift={shift_t:.4f}m/{shift_r:.3f}deg",
                flush=True,
            )
        _write_constraints(m3_csv, fieldnames, m3_rows)
        workspace.clear_caches(include_shared_frames=True)
        result = {
            "dataset": dataset,
            "phase": "measurement_and_pgo_no_truth",
            "truth_opened": False,
            "plain_measurement_source": plain_measurement_source,
            "m3_config": asdict(config),
            "frozen_constraints": str(frozen_csv.resolve()),
            "frozen_constraints_sha256": sha256_file(frozen_csv),
            "plain_constraints_sha256": sha256_file(plain_csv),
            "m3_constraints_sha256": sha256_file(m3_csv),
            "pair_count": len(plain_rows),
            "m3_used_count": sum(item["used_m3"] for item in measurements),
            "fallback_count": sum(not item["used_m3"] for item in measurements),
            "measurements": measurements,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    plain_report = _run_optimizer(
        session=session,
        constraints=plain_csv,
        output=output / "pgo_plain",
        resume=resume,
    )
    m3_report = _run_optimizer(
        session=session,
        constraints=m3_csv,
        output=output / "pgo_m3",
        resume=resume,
    )
    result["pgo"] = {
        "plain_report": str((output / "pgo_plain/manual_loop_report.json").resolve()),
        "m3_report": str((output / "pgo_m3/manual_loop_report.json").resolve()),
        "plain_trajectory": str((output / "pgo_plain/optimized_poses_tum.txt").resolve()),
        "m3_trajectory": str((output / "pgo_m3/optimized_poses_tum.txt").resolve()),
        "plain_robust_weights": plain_report.get("robust_weights"),
        "m3_robust_weights": m3_report.get("robust_weights"),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _truth_for_session(session: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = load_tum_trajectory(session / "optimized_poses_tum.txt")
    if manifest["truth_kind"] == "se3":
        frame_config = json.loads(FRAME_CONFIG.read_text(encoding="utf-8"))
        role, _ = manifest_keyframe_frame_role(manifest, frame_config)
        return interpolated_truth_in_keyframe_frame(
            trajectory.timestamps,
            Path(manifest["truth"]),
            manifest["dataset"],
            frame_config,
            keyframe_frame_role=role,
        )
    return nearest_truth(
        trajectory.timestamps,
        Path(manifest["truth"]),
        manifest["truth_kind"],
    )


def evaluation_phase(
    *,
    dataset: str,
    session: Path,
    output: Path,
    no_truth_result: dict,
) -> dict:
    """Attach truth metrics only after both frozen PGO arms have completed."""
    manifest = json.loads((session / "input_manifest.json").read_text(encoding="utf-8"))
    truth, valid, orientation_valid = _truth_for_session(session, manifest)
    measurement_rows = []
    for item in no_truth_result["measurements"]:
        source_id = item["source_id"]
        target_id = item["target_id"]
        row = {
            "row_index": item["row_index"],
            "source_id": source_id,
            "target_id": target_id,
            "truth_available": bool(valid[source_id] and valid[target_id]),
            "orientation_truth_available": bool(
                orientation_valid[source_id] and orientation_valid[target_id]
            ),
        }
        if row["truth_available"] and manifest["truth_kind"] == "se3":
            plain = np.asarray(item["plain_transform_target_source"], dtype=float)
            selected = np.asarray(item["selected_transform_target_source"], dtype=float)
            plain_t, plain_r = target_to_source_truth_error(
                plain, truth[source_id], truth[target_id]
            )
            m3_t, m3_r = target_to_source_truth_error(
                selected, truth[source_id], truth[target_id]
            )
            row.update({
                "plain_translation_error_m": plain_t,
                "m3_translation_error_m": m3_t,
                "translation_error_change_m": m3_t - plain_t,
                "plain_rotation_error_deg": plain_r,
                "m3_rotation_error_deg": m3_r,
                "rotation_error_change_deg": m3_r - plain_r,
            })
        measurement_rows.append(row)

    odometry_ate = trajectory_metric(
        dataset, session / "optimized_poses_tum.txt", manifest
    )
    plain_ate = trajectory_metric(
        dataset, output / "pgo_plain/optimized_poses_tum.txt", manifest
    )
    m3_ate = trajectory_metric(
        dataset, output / "pgo_m3/optimized_poses_tum.txt", manifest
    )
    report = {
        "dataset": dataset,
        "phase": "posthoc_truth_evaluation",
        "measurement_phase_truth_opened": False,
        "odometry_ate_m": odometry_ate,
        "plain_pgo_ate_m": plain_ate,
        "m3_pgo_ate_m": m3_ate,
        "m3_minus_plain_ate_m": (
            None if plain_ate is None or m3_ate is None else m3_ate - plain_ate
        ),
        "m3_relative_ate_change": (
            None
            if plain_ate in (None, 0.0) or m3_ate is None
            else (m3_ate - plain_ate) / plain_ate
        ),
        "loop_truth": measurement_rows,
    }
    (output / "evaluation_with_truth.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("frozen_profile_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--plain-measurement-source",
        choices=("gicp_ledger", "frozen_csv"),
        default="gicp_ledger",
        help=(
            "Use raw ordinary GICP from the immutable proposal ledger, or the "
            "possibly remeasured transform already stored in the frozen CSV."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rebuild_root = args.rebuild_root.expanduser().resolve()
    frozen_root = args.frozen_profile_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    no_truth_results: dict[str, dict] = {}
    for dataset in args.datasets:
        session = rebuild_root / dataset
        frozen_csv = frozen_root / dataset / "candidate_constraints.csv"
        if not session.is_dir() or not frozen_csv.is_file():
            raise FileNotFoundError(f"Missing frozen input for {dataset}")
        no_truth_results[dataset] = measurement_and_pgo_phase(
            dataset=dataset,
            session=session,
            frozen_csv=frozen_csv,
            output=output_root / dataset,
            resume=args.resume,
            plain_measurement_source=args.plain_measurement_source,
        )

    # This loop is intentionally separate: no truth path is opened until every
    # requested dataset has finished both measurement arms and PGO.
    evaluations = {}
    for dataset in args.datasets:
        evaluations[dataset] = evaluation_phase(
            dataset=dataset,
            session=rebuild_root / dataset,
            output=output_root / dataset,
            no_truth_result=no_truth_results[dataset],
        )

    relative_changes = [
        item["m3_relative_ate_change"]
        for item in evaluations.values()
        if item["m3_relative_ate_change"] is not None
    ]
    summary = {
        "schema_version": 1,
        "protocol": {
            "name": "TRO_M3_frozen_pair_measurement_ablation",
            "selection": "none; TRO parameters frozen before truth evaluation",
            "truth_blinding": (
                "all M3 measurements and both PGO arms completed before truth was opened"
            ),
            "pair_policy": "identical frozen pairs; invalid M3 falls back to plain",
            "plain_measurement_source": args.plain_measurement_source,
            "pgo_profile": {
                "mode": "gnc_tls",
                "sigma_t_m": math.sqrt(1.6),
                "sigma_r_deg": 0.5,
                "manual_information_scale": 0.1,
                "correlation_window": 0,
                "cluster_budget": 0.0,
            },
        },
        "datasets": evaluations,
        "aggregate": {
            "dataset_count": len(evaluations),
            "m3_better_count": sum(value < 0.0 for value in relative_changes),
            "m3_worse_count": sum(value > 0.0 for value in relative_changes),
            "mean_relative_ate_change": (
                float(np.mean(relative_changes)) if relative_changes else None
            ),
            "max_relative_ate_increase": (
                float(max(relative_changes)) if relative_changes else None
            ),
        },
    }
    summary_path = output_root / "m3_measurement_ablation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "m3_measurement_ablation.json.sha256").write_text(
        sha256_file(summary_path) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
