#!/usr/bin/env python3
"""Probe whether symmetric source submaps repair accepted initial-loop registrations.

This is a bounded diagnostic, not a production gate.  It starts every arm from
the exact frozen GICP measurement in the immutable proposal ledger, changes
only the number of neighbouring source keyframes, and evaluates the resulting
relative transform against public truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
GHOST_EVAL = HERE.parents[1] / "ghostloop_eval"
for extra in (GUI, GHOST_EVAL):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_icra2027_rebuild import (  # noqa: E402
    interpolated_truth_in_keyframe_frame,
)
from manual_loop_closure.registration import (  # noqa: E402
    RegistrationConfig,
    RegistrationWorkspace,
)
from manual_loop_closure.trajectory_io import load_tum_trajectory  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_from_record(record: dict) -> np.ndarray:
    item = record["gicp"]["transform_target_source"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        item["quaternion_xyzw"]
    ).as_matrix()
    transform[:3, 3] = item["translation_m"]
    return transform


def truth_error(measurement: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    error = np.linalg.inv(expected) @ measurement
    return (
        float(np.linalg.norm(error[:3, 3])),
        float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude())),
    )


def admitted_records(session: Path, active_pairs: set[tuple[int, int]]) -> list[dict]:
    ledger_paths = sorted((session / "proposal_ledgers").glob("*.jsonl"))
    if len(ledger_paths) != 1:
        raise ValueError(f"expected one ledger in {session}")
    output = []
    with ledger_paths[0].open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            pair = (int(record.get("source_id", -1)), int(record.get("target_id", -1)))
            if (
                record.get("source") == "stage0_descriptor"
                and record.get("decision") == "admitted_probationary"
                and pair in active_pairs
                and record.get("gicp")
            ):
                output.append(record)
    return output


def run(root: Path, output_dir: Path, windows: list[int], frame_config_path: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame_config = json.loads(frame_config_path.read_text(encoding="utf-8"))
    truth_rows = list(csv.DictReader(
        (root / "c2_candidate_source_data.csv").open(encoding="utf-8")
    ))
    rows: list[dict] = []
    dataset_reports = {}
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = session / "input_manifest.json"
        summary_path = session / "auto_repair_summary.json"
        if not (manifest_path.is_file() and summary_path.is_file()):
            continue
        dataset = session.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_truth = [
            item for item in truth_rows
            if item["dataset"] == dataset
            and item["active_final"].lower() == "true"
        ]
        truth_by_pair = {
            (int(item["source_id"]), int(item["target_id"])): item
            for item in selected_truth
        }
        records = admitted_records(session, set(truth_by_pair))
        trajectory = load_tum_trajectory(session / "optimized_poses_tum.txt")
        truth_poses, valid, _ = interpolated_truth_in_keyframe_frame(
            trajectory.timestamps, Path(manifest["truth"]),
            manifest["dataset"], frame_config,
        )
        workspace = RegistrationWorkspace(
            Path(manifest["keyframes"]["resolved_path"]), trajectory
        )
        base_config = RegistrationConfig(**summary["registration"]["near"])
        for record in records:
            source_id = int(record["source_id"])
            target_id = int(record["target_id"])
            frozen = transform_from_record(record)
            expected = (
                np.linalg.inv(truth_poses[target_id]) @ truth_poses[source_id]
                if valid[source_id] and valid[target_id] else None
            )
            frozen_translation_error, frozen_rotation_error = (
                truth_error(frozen, expected)
                if expected is not None else (float("nan"), float("nan"))
            )
            delta = (
                np.linalg.inv(trajectory.transforms_world_sensor[source_id])
                @ trajectory.transforms_world_sensor[target_id]
                @ frozen
            )
            truth_row = truth_by_pair[(source_id, target_id)]
            for window in windows:
                try:
                    result = workspace.run_gicp(
                        source_id=source_id,
                        target_id=target_id,
                        delta_transform_local=delta,
                        config=replace(base_config, source_window=window),
                    )
                    measurement = result.transform_target_source_final
                    translation_error, rotation_error = (
                        truth_error(measurement, expected)
                        if expected is not None else (float("nan"), float("nan"))
                    )
                    move = np.linalg.inv(frozen) @ measurement
                    row = {
                        "dataset": dataset,
                        "proposal_id": record["proposal_id"],
                        "source_id": source_id,
                        "target_id": target_id,
                        "window": window,
                        "truth_correct_frozen_1m_5deg": int(
                            truth_row["truth_correct"].lower() == "true"
                        ),
                        "frozen_translation_error_m": frozen_translation_error,
                        "frozen_rotation_error_deg": frozen_rotation_error,
                        "fitness": result.fitness,
                        "rmse_m": result.inlier_rmse,
                        "movement_from_frozen_m": float(np.linalg.norm(move[:3, 3])),
                        "movement_from_frozen_deg": float(np.degrees(
                            Rotation.from_matrix(move[:3, :3]).magnitude()
                        )),
                        "translation_error_m": translation_error,
                        "rotation_error_deg": rotation_error,
                        "truth_correct_1m_5deg": int(
                            translation_error <= 1.0 and rotation_error <= 5.0
                        ),
                        "valid": 1,
                        "reason": "completed",
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "dataset": dataset,
                        "proposal_id": record["proposal_id"],
                        "source_id": source_id,
                        "target_id": target_id,
                        "window": window,
                        "truth_correct_frozen_1m_5deg": int(
                            truth_row["truth_correct"].lower() == "true"
                        ),
                        "frozen_translation_error_m": frozen_translation_error,
                        "frozen_rotation_error_deg": frozen_rotation_error,
                        "fitness": float("nan"),
                        "rmse_m": float("nan"),
                        "movement_from_frozen_m": float("nan"),
                        "movement_from_frozen_deg": float("nan"),
                        "translation_error_m": float("nan"),
                        "rotation_error_deg": float("nan"),
                        "truth_correct_1m_5deg": 0,
                        "valid": 0,
                        "reason": type(exc).__name__,
                    }
                rows.append(row)
        dataset_reports[dataset] = {
            "active_candidates": len(records),
            "truth_matched_poses": int(valid.sum()),
            "trajectory_sha256": sha256_file(session / "optimized_poses_tum.txt"),
            "ledger_sha256": sha256_file(next((session / "proposal_ledgers").glob("*.jsonl"))),
        }

    csv_path = output_dir / "multiframe_registration_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    window_summary = {}
    for window in windows:
        selected = [row for row in rows if row["window"] == window and row["valid"]]
        window_summary[str(window)] = {
            "completed": len(selected),
            "truth_correct_1m_5deg": int(sum(
                row["truth_correct_1m_5deg"] for row in selected
            )),
            "median_translation_error_m": float(np.median([
                row["translation_error_m"] for row in selected
            ])) if selected else None,
            "median_rotation_error_deg": float(np.median([
                row["rotation_error_deg"] for row in selected
            ])) if selected else None,
            "median_movement_from_frozen_m": float(np.median([
                row["movement_from_frozen_m"] for row in selected
            ])) if selected else None,
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": (
            "exact frozen GICP measurement as initialization; near-tier target "
            "configuration fixed; only symmetric source-window radius changes"
        ),
        "windows": windows,
        "dataset_reports": dataset_reports,
        "window_summary": window_summary,
        "source_data": {
            "csv": csv_path.name,
            "csv_sha256": sha256_file(csv_path),
            "script_sha256": sha256_file(HERE),
            "frame_config_sha256": sha256_file(frame_config_path),
        },
    }
    (output_dir / "multiframe_registration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[0, 2, 4, 8, 12]
    )
    args = parser.parse_args()
    report = run(
        args.c2_root.resolve(), args.output_dir.resolve(),
        sorted(set(args.windows)), args.frame_config.resolve(),
    )
    print(json.dumps(report["window_summary"], indent=2))


if __name__ == "__main__":
    main()
