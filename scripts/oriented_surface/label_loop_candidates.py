#!/usr/bin/env python3
"""Truth-label loop measurements and freeze an oracle set for factor isolation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from calibrate_factor_information import (
    factor_error_g2o,
    interpolated_truth,
    truth_body_to_keyframe_from_manifest,
    tum_data,
)
from generate_proximity_loop_candidates import CSV_FIELDS
from rebuild_oriented_pose_graph import sha256_file, transform_from_fields


def label(
    session: Path,
    candidates_csv: Path,
    frame_config_path: Path,
    output_dir: Path,
    *,
    max_translation_error_m: float,
    max_rotation_error_deg: float,
    max_truth_dt_s: float,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest = json.loads((session / "input_manifest.json").read_text())
    if manifest.get("truth_kind") != "se3":
        raise ValueError("loop factor isolation requires SE(3) truth")
    timestamps, _ = tum_data(session / "optimized_poses_tum.txt")
    truth, truth_valid = interpolated_truth(
        timestamps, Path(manifest["truth"]), max_truth_dt_s
    )
    frame_config = json.loads(frame_config_path.read_text(encoding="utf-8"))
    truth_to_keyframe, frame_resolution = truth_body_to_keyframe_from_manifest(
        manifest, frame_config
    )
    truth = truth @ truth_to_keyframe
    rows = list(csv.DictReader(candidates_csv.open(encoding="utf-8")))
    labels = []
    accepted_rows = []
    for row in rows:
        target_id, source_id = int(row["target_id"]), int(row["source_id"])
        matched = bool(truth_valid[target_id] and truth_valid[source_id])
        translation_error = rotation_error = None
        correct = False
        if matched:
            transform = transform_from_fields(
                [float(row[key]) for key in ("tx", "ty", "tz")],
                [float(row[key]) for key in ("qx", "qy", "qz", "qw")],
            )
            error = factor_error_g2o(
                transform, truth[target_id], truth[source_id]
            )
            translation_error = float(np.linalg.norm(error[:3]))
            rotation_error = float(np.degrees(np.linalg.norm(error[3:])))
            correct = (
                translation_error <= max_translation_error_m
                and rotation_error <= max_rotation_error_deg
            )
        labels.append({
            "dataset": manifest["dataset"],
            "target_id": target_id,
            "source_id": source_id,
            "truth_matched": matched,
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "truth_correct": correct,
        })
        if correct:
            accepted_rows.append(row)
    label_path = output_dir / "loop_truth_labels.csv"
    with label_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]) if labels else ["dataset"])
        writer.writeheader()
        writer.writerows(labels)
    oracle_path = output_dir / "oracle_true_loop_candidates.csv"
    with oracle_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(accepted_rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "oracle loop set isolates factor construction; not a deployable proposal policy",
        "dataset": manifest["dataset"],
        "thresholds": {
            "translation_m": max_translation_error_m,
            "rotation_deg": max_rotation_error_deg,
            "truth_association_s": max_truth_dt_s,
        },
        "counts": {
            "candidate": len(rows),
            "truth_matched": sum(row["truth_matched"] for row in labels),
            "truth_correct": len(accepted_rows),
            "truth_wrong": sum(row["truth_matched"] and not row["truth_correct"] for row in labels),
        },
        "inputs": {
            "candidates_sha256": sha256_file(candidates_csv),
            "truth_sha256": sha256_file(Path(manifest["truth"])),
            "frame_config_sha256": sha256_file(frame_config_path),
            "frame_resolution": frame_resolution,
        },
        "outputs": {
            "labels_sha256": sha256_file(label_path),
            "oracle_csv_sha256": sha256_file(oracle_path),
        },
    }
    (output_dir / "loop_truth_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-threshold", type=float, default=1.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--max-truth-dt", type=float, default=0.05)
    args = parser.parse_args()
    report = label(
        args.session.expanduser().resolve(),
        args.candidates_csv.expanduser().resolve(),
        args.frame_config.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        max_translation_error_m=args.translation_threshold,
        max_rotation_error_deg=args.rotation_threshold_deg,
        max_truth_dt_s=args.max_truth_dt,
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
