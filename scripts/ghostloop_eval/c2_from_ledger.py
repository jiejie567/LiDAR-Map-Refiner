#!/usr/bin/env python3
"""Aggregate truth-labelled descriptor proposals for the C2 audit benchmark.

The input runs are produced with a recall-first descriptor pool and the normal
headless pipeline. Each registered proposal carries its measured transform,
odometry-budget occupancy, trial-PGO residuals, and final decision in the
hash-chained ledger. This script derives threshold sensitivities without
re-running registration or silently dropping failed audit outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_icra2027_rebuild import evaluate, sha256_file
from proposal_ledger import ProposalLedger


def is_truth_correct(
    row: dict, translation_m: float, rotation_deg: float,
    measurement_prefix: str = "",
) -> bool:
    prefix = f"{measurement_prefix}_" if measurement_prefix else ""
    translation = row.get(f"{prefix}relative_translation_error_m")
    rotation = row.get(f"{prefix}relative_rotation_error_deg")
    return (
        translation is not None and rotation is not None
        and translation <= translation_m and rotation <= rotation_deg
    )


def metrics(rows: list[dict], accept) -> dict:
    key = "truth_correct"
    truth = [
        row for row in rows
        if row.get("gt_available") and row.get(key) is not None
    ]
    correct = [row for row in truth if row[key]]
    wrong = [row for row in truth if not row[key]]
    accepted = [row for row in truth if accept(row)]
    true_accept = sum(row[key] for row in accepted)
    false_accept = len(accepted) - true_accept
    return {
        "registered_candidates": len(truth),
        "truth_correct": len(correct),
        "truth_wrong": len(wrong),
        "accepted": len(accepted),
        "true_accepted": true_accept,
        "false_accepted": false_accept,
        "precision": true_accept / len(accepted) if accepted else 1.0,
        "recall": true_accept / len(correct) if correct else None,
        "false_accept_rate": false_accept / len(wrong) if wrong else None,
        "catastrophic_over_5m": sum(
            row.get("gicp_relative_translation_error_m") is not None
            and row["gicp_relative_translation_error_m"] > 5.0
            for row in accepted
        ),
    }


def proposal_key(row: dict) -> tuple[str, int, int]:
    """Return the stable identity used to detect development-pool reuse."""
    return (
        str(row["dataset"]),
        int(row["target_id"]),
        int(row["source_id"]),
    )


def policy_metrics(rows: list[dict], policies: dict) -> dict:
    return {
        name: metrics(rows, predicate)
        for name, predicate in policies.items()
    }


def descriptor_proposal_keys(root: Path) -> set[tuple[str, int, int]]:
    """Read every proposed descriptor pair, including failed registrations."""
    keys: set[tuple[str, int, int]] = set()
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = session / "auto_repair_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ledger_path = Path(summary["proposal_ledger"]["path"])
        for row in ProposalLedger.verify(ledger_path):
            if row.get("source") != "stage0_descriptor":
                continue
            keys.add((session.name, int(row["target_id"]), int(row["source_id"])))
    return keys


def comparison_metrics(policy_results: dict) -> dict:
    """Summarize the two preregistered C2 comparison quantities."""
    gicp = policy_results["gicp_only"]
    full = policy_results["full_audit"]
    if gicp["false_accept_rate"] is not None and gicp["false_accept_rate"] > 0:
        false_accept_reduction = (
            gicp["false_accept_rate"] - full["false_accept_rate"]
        ) / gicp["false_accept_rate"]
    else:
        false_accept_reduction = None
    if gicp["recall"] is not None and full["recall"] is not None:
        recall_drop = 100 * (gicp["recall"] - full["recall"])
    else:
        recall_drop = None
    return {
        "false_accept_reduction_vs_gicp": false_accept_reduction,
        "recall_drop_percentage_points": recall_drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("--translation", type=float, default=1.0)
    parser.add_argument("--rotation", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Write derived evidence outside the immutable rebuild root",
    )
    parser.add_argument(
        "--development-root", type=Path,
        help=(
            "Earlier candidate-pool root used to select thresholds. When set, "
            "the report separates repeated proposal pairs from genuinely new "
            "pairs; the expanded pool is never mislabeled as wholly held out."
        ),
    )
    parser.add_argument(
        "--baseline-manifest", type=Path,
        help="Explicit Huber/GNC baseline manifest stored outside the run root",
    )
    args = parser.parse_args()
    root = args.rebuild_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(root)
    development_keys: set[tuple[str, int, int]] = set()
    if args.development_root:
        development_root = args.development_root.expanduser().resolve()
        development_keys = descriptor_proposal_keys(development_root)
    baseline_path = (
        args.baseline_manifest.expanduser().resolve()
        if args.baseline_manifest else root / "c2_baseline_manifest.json"
    )
    baselines = (
        json.loads(baseline_path.read_text()) if baseline_path.is_file() else {}
    )
    gnc_weights = {}
    for dataset, arms in baselines.items():
        arm = arms.get("gnc_tls", {})
        for proposal, weight in zip(
            arm.get("candidate_proposal_ids", []),
            arm.get("candidate_weights") or [],
        ):
            gnc_weights[(dataset, proposal)] = float(weight)
    rows = []
    for row in report["proposal_rows"]:
        if row["source"] != "stage0_descriptor":
            continue
        if (
            row.get("relative_translation_error_m") is None
            or row.get("relative_rotation_error_deg") is None
        ):
            continue
        row = dict(row)
        row["gnc_tls_weight"] = gnc_weights.get(
            (row["dataset"], row["proposal_id"])
        )
        for prefix in ("gicp", "factor"):
            translation_value = row.get(
                f"{prefix}_relative_translation_error_m"
            )
            rotation_value = row.get(f"{prefix}_relative_rotation_error_deg")
            row[f"truth_correct_{prefix}"] = (
                None if translation_value is None or rotation_value is None
                else is_truth_correct(
                    row, args.translation, args.rotation, prefix
                )
            )
        row["truth_correct"] = row["truth_correct_gicp"]
        row["evaluation_split"] = (
            "development_overlap"
            if proposal_key(row) in development_keys
            else "novel_pair"
        )
        rows.append(row)

    policies = {
        # This ledger begins after the GICP quality gate, so accepting every
        # registered row is exactly the GICP-only baseline for this pool.
        "gicp_only": lambda row: bool(row["gicp_gate_passed"]),
        "gicp_huber_pgo": lambda row: bool(row["gicp_gate_passed"]),
        "gnc_tls": lambda row: (
            row["gnc_tls_weight"] is not None
            and row["gnc_tls_weight"] >= 0.5
        ),
        "odometry_budget": lambda row: (
            bool(row["gicp_gate_passed"])
            and row["odometry_budget_occupancy"] is not None
            and row["odometry_budget_occupancy"] <= 1.0
        ),
        "gravity_admission": lambda row: (
            bool(row["gicp_gate_passed"])
            and row["odometry_budget_occupancy"] is not None
            and row["odometry_budget_occupancy"] <= 1.0
            and row.get("gravity_gate_passed_1deg") is True
        ),
        # active_final is the only honest end-to-end policy in a run whose
        # early gravity rejection prevents later probation fields from being
        # observed.  Do not reuse it as a fictitious no-gravity ablation.
        "full_audit": lambda row: bool(row["active_final"]),
    }
    summary = {
        "schema_version": 2,
        "truth_threshold": {
            "translation_m": args.translation,
            "rotation_deg": args.rotation,
        },
        "dataset_count": len({row["dataset"] for row in rows}),
        "policies": policy_metrics(rows, policies),
        "trajectory_baselines": baselines,
        "baseline_manifest": {
            "path": str(baseline_path),
            "available": baseline_path.is_file(),
            "sha256": sha256_file(baseline_path) if baseline_path.is_file() else None,
        },
        "gravity_gate": {
            "max_error_deg": 1.0,
            "definition": (
                "angle(R_target_source * up_source, up_target)"
            ),
            "selection_status": (
                "fixed after exploratory v1 pool; repeated and novel pairs are "
                "reported separately"
            ),
        },
    }
    if args.development_root:
        split_rows = {
            "development_overlap": [
                row for row in rows
                if row["evaluation_split"] == "development_overlap"
            ],
            "novel_pairs": [
                row for row in rows
                if row["evaluation_split"] == "novel_pair"
            ],
        }
        summary["development_pool"] = {
            "root": str(development_root),
            "proposal_pair_count": len(development_keys),
            "interpretation": (
                "Only novel_pairs estimates pair-level generalization. It is "
                "not an independent sequence-level holdout."
            ),
        }
        summary["evaluation_splits"] = {}
        for split_name, selected in split_rows.items():
            split_policies = policy_metrics(selected, policies)
            summary["evaluation_splits"][split_name] = {
                "candidate_rows": len(selected),
                "dataset_count": len({row["dataset"] for row in selected}),
                "policies": split_policies,
                **comparison_metrics(split_policies),
            }
    sensitivity = {}
    for translation in (0.5, 1.0, 2.0):
        for rotation in (2.0, 5.0, 10.0):
            labelled = []
            for row in rows:
                item = dict(row)
                for prefix in ("gicp", "factor"):
                    t_value = item.get(f"{prefix}_relative_translation_error_m")
                    r_value = item.get(f"{prefix}_relative_rotation_error_deg")
                    item[f"truth_correct_{prefix}"] = (
                        None if t_value is None or r_value is None
                        else is_truth_correct(item, translation, rotation, prefix)
                    )
                item["truth_correct"] = item["truth_correct_gicp"]
                labelled.append(item)
            key = f"{translation:.1f}m_{rotation:.0f}deg"
            sensitivity[key] = {
                name: metrics(labelled, predicate)
                for name, predicate in policies.items()
            }
    summary["threshold_sensitivity"] = sensitivity
    gicp = summary["policies"]["gicp_only"]
    full = summary["policies"]["full_audit"]
    summary.update(comparison_metrics(summary["policies"]))
    summary["c2_data_gate"] = {
        "at_least_four_datasets": summary["dataset_count"] >= 4,
        "at_least_100_registered": gicp["registered_candidates"] >= 100,
        "at_least_20_wrong": gicp["truth_wrong"] >= 20,
    }
    summary["c2_method_gate"] = {
        "false_accept_reduction_at_least_50pct": (
            summary["false_accept_reduction_vs_gicp"] is not None
            and summary["false_accept_reduction_vs_gicp"] >= 0.50
        ),
        "recall_drop_at_most_10pp": (
            summary["recall_drop_percentage_points"] is not None
            and summary["recall_drop_percentage_points"] <= 10.0
        ),
        "no_catastrophic_loop_over_5m": (
            full["catastrophic_over_5m"] == 0
        ),
    }
    dataset_wrong = {
        dataset: sum(
            row["dataset"] == dataset and not row["truth_correct"] for row in rows
        )
        for dataset in {row["dataset"] for row in rows}
    }
    comparable = []
    substantially_better = []
    for dataset, arms in baselines.items():
        gnc_ate = (arms.get("gnc_tls") or {}).get("ate_m")
        full_ate = (arms.get("full_audit") or {}).get("ate_m")
        if gnc_ate is None or full_ate is None:
            continue
        comparable.append(
            full_ate <= gnc_ate + max(0.02, 0.02 * gnc_ate)
        )
        if dataset_wrong.get(dataset, 0) and full_ate <= 0.90 * gnc_ate:
            substantially_better.append(dataset)
    summary["c2_method_gate"]["ate_not_worse_than_gnc_tls"] = (
        bool(comparable) and all(comparable)
    )
    summary["c2_method_gate"]["ate_substantially_better_on_one_aliased_sequence"] = (
        bool(substantially_better)
    )
    summary["substantially_better_than_gnc_datasets"] = substantially_better
    factor_rows = [
        row for row in rows if row.get("truth_correct_factor") is not None
    ]
    summary["factor_remeasurement"] = {
        "evaluated": len(factor_rows),
        "gicp_correct": sum(bool(row["truth_correct_gicp"]) for row in factor_rows),
        "factor_correct": sum(bool(row["truth_correct_factor"]) for row in factor_rows),
        "corrected_by_factor": sum(
            not row["truth_correct_gicp"] and row["truth_correct_factor"]
            for row in factor_rows
        ),
        "degraded_by_factor": sum(
            row["truth_correct_gicp"] and not row["truth_correct_factor"]
            for row in factor_rows
        ),
        "note": (
            "C2 proposal precision/recall is labelled on the frozen GICP "
            "measurement; oriented factor remeasurement is reported separately."
        ),
    }
    summary["passed"] = (
        all(summary["c2_data_gate"].values())
        and all(summary["c2_method_gate"].values())
    )
    json_path = output_dir / "c2_source_data.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "c2_candidate_source_data.csv"
    fields = list(rows[0]) if rows else ["dataset", "proposal_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "c2_evidence_manifest.json").write_text(
        json.dumps({
            json_path.name: sha256_file(json_path),
            csv_path.name: sha256_file(csv_path),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
