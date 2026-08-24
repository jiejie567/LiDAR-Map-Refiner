#!/usr/bin/env python3
"""Freeze the decision on ICP-remeasured pose-graph factors.

The report deliberately distinguishes three questions that are easy to blur:
replacing odometry measurements, reweighting odometry measurements, and
remeasuring only loop factors.  A PGO module is promoted to a paper
contribution only if it improves over the controlled original/loop-only arms
and the direction of the gain repeats across datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset", "controlled_inputs", "variants"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    return report


def variants_by_name(report: dict) -> dict[str, dict]:
    return {row["variant"]: row for row in report["variants"]}


def ate(report: dict, variant: str) -> float:
    return float(variants_by_name(report)[variant]["position_ate_rmse_m"])


def assert_same_inputs(left: dict, right: dict) -> None:
    for key in ("pose_graph_sha256", "trajectory_sha256", "loop_csv_sha256"):
        if left["controlled_inputs"][key] != right["controlled_inputs"][key]:
            raise ValueError(f"Non-controlled comparison: {key} differs")


def relative_change(candidate: float, reference: float) -> float:
    return (candidate - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replacement_report", type=Path)
    parser.add_argument("reweight_report", type=Path)
    parser.add_argument("loop_only_report", type=Path)
    parser.add_argument("calibrated_reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = [
        args.replacement_report,
        args.reweight_report,
        args.loop_only_report,
        *args.calibrated_reports,
    ]
    paths = [path.expanduser().resolve() for path in paths]
    replacement, reweight, loop_only, *calibrated = map(load_report, paths)
    assert_same_inputs(replacement, reweight)
    assert_same_inputs(reweight, loop_only)

    replacement_baseline = ate(replacement, "original_graph")
    replacement_results = {
        name: {
            "ate_m": ate(replacement, name),
            "relative_change_vs_original": relative_change(
                ate(replacement, name), replacement_baseline
            ),
        }
        for name in (
            "icp_isotropic", "icp_anisotropic", "oriented_anisotropic"
        )
    }
    sequential_oriented = ate(reweight, "oriented_anisotropic_reweight")
    loop_oriented = ate(loop_only, "loop_oriented_anisotropic")

    calibrated_rows = []
    for report in calibrated:
        isotropic = ate(report, "loop_icp_isotropic")
        anisotropic = ate(report, "loop_icp_anisotropic")
        oriented = ate(report, "loop_oriented_anisotropic")
        calibrated_rows.append({
            "dataset": report["dataset"],
            "original_graph_ate_m": ate(report, "original_graph"),
            "loop_icp_isotropic_ate_m": isotropic,
            "loop_icp_anisotropic_ate_m": anisotropic,
            "loop_oriented_anisotropic_ate_m": oriented,
            "oriented_relative_change_vs_icp_anisotropic": relative_change(
                oriented, anisotropic
            ),
        })

    all_replacements_worse = all(
        row["ate_m"] >= replacement_baseline
        for row in replacement_results.values()
    )
    reweight_beats_loop_only = sequential_oriented < loop_oriented
    oriented_changes = [
        row["oriented_relative_change_vs_icp_anisotropic"]
        for row in calibrated_rows
    ]
    consistent_oriented_gain = bool(oriented_changes) and all(
        change < 0.0 for change in oriented_changes
    )
    promoted = (
        not all_replacements_worse
        and reweight_beats_loop_only
        and consistent_oriented_gain
    )

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "question": (
            "Does ICP remeasurement plus geometry-derived information provide "
            "a repeatable PGO contribution beyond loop measurement alone?"
        ),
        "source_reports": {
            str(path): sha256_file(path) for path in paths
        },
        "controlled_input_hashes": replacement["controlled_inputs"],
        "sequential_measurement_replacement": {
            "original_graph_ate_m": replacement_baseline,
            "variants": replacement_results,
            "all_replacement_variants_worse": all_replacements_worse,
        },
        "sequential_information_reweighting": {
            "oriented_reweight_ate_m": sequential_oriented,
            "loop_only_oriented_ate_m": loop_oriented,
            "relative_change_vs_loop_only": relative_change(
                sequential_oriented, loop_oriented
            ),
            "beats_loop_only": reweight_beats_loop_only,
        },
        "calibrated_loop_information": calibrated_rows,
        "gates": {
            "replacement_improves_original": not all_replacements_worse,
            "sequential_reweight_beats_loop_only": reweight_beats_loop_only,
            "oriented_loop_gain_repeats_across_datasets": consistent_oriented_gain,
        },
        "promote_as_paper_contribution": promoted,
        "recommended_role": (
            "paper contribution" if promoted
            else "implementation option and negative/auxiliary ablation only"
        ),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
