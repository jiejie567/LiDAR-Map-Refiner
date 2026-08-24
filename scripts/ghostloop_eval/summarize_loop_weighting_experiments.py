#!/usr/bin/env python3
"""Freeze the exploratory loop-weighting experiments into one decision report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_icra2027_rebuild import evaluate  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-ablation", required=True, type=Path)
    parser.add_argument("--mode-ablation", required=True, type=Path)
    parser.add_argument("--correlation-stress", required=True, type=Path)
    parser.add_argument("--terminal-ablation", required=True, type=Path)
    parser.add_argument("--legacy-c1-root", required=True, type=Path)
    parser.add_argument("--calibrated-c1-root", required=True, type=Path)
    parser.add_argument("--legacy-c2-baselines", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    scale = load(args.scale_ablation)
    mode = load(args.mode_ablation)
    correlation = load(args.correlation_stress)
    terminal = load(args.terminal_ablation)
    legacy_c2 = load(args.legacy_c2_baselines)
    old_c1 = evaluate(args.legacy_c1_root)["dataset_reports"]["ntu"]
    new_c1 = evaluate(args.calibrated_c1_root)["dataset_reports"]["ntu"]

    selected_name = scale["best_profile"]
    selected = scale["profiles"][selected_name]
    candidate_pool = {}
    for tag, arms in selected["datasets"].items():
        old = legacy_c2[tag]["gnc_tls"]["ate_m"]
        new = arms["all"]["ate_m"]
        odometry = scale["odometry_ate_m"][tag]
        candidate_pool[tag] = {
            "odometry_ate_m": odometry,
            "legacy_gnc_ate_m": old,
            "calibrated_ate_m": new,
            "calibrated_vs_odometry_percent": 100.0 * (new / odometry - 1.0),
            "calibrated_minus_legacy_m": new - old,
            "candidate_count": arms["all"]["candidate_count"],
            "gnc_unique_weights": sorted(set(
                round(float(value), 12)
                for value in (arms["all"].get("candidate_weights") or [])
            )),
        }

    corr_budget = correlation["profiles"][
        "gnc_tls_t1p26491_r0p5deg_s0p1_w1_b1"
    ]["all_score"]["ate_ratio_by_dataset"]
    single = selected["all_score"]["ate_ratio_by_dataset"]
    correlation_max_difference = max(
        abs(float(corr_budget[tag]) - float(single[tag])) for tag in single
    )

    terminal_best_name = terminal["best_profile"]
    terminal_best = terminal["profiles"][terminal_best_name]
    fixed_ntu = terminal_best["datasets"]["ntu"]
    mode_scores = {
        name: item["all_score"]["geometric_mean_ate_ratio"]
        for name, item in mode["profiles"].items()
    }
    payload = {
        "status": "exploratory_development_result_not_held_out",
        "recommended_profile": {
            "name": "calibrated_correlated_gnc",
            "solver": "gnc_tls",
            "sigma_t_m": 1.264911064,
            "sigma_r_deg": 0.5,
            "manual_information_scale": 0.1,
            "correlation_window_keyframes": 30,
            "cluster_information_budget": 1.0,
            "legacy_profile_available": True,
        },
        "candidate_pool_result": candidate_pool,
        "candidate_pool_geometric_mean_ate_ratio": selected[
            "all_score"
        ]["geometric_mean_ate_ratio"],
        "solver_mode_geometric_mean_ratios": mode_scores,
        "gnc_rejected_candidate_count": sum(
            sum(weight < 0.5 for weight in (arms["all"].get("candidate_weights") or []))
            for arms in selected["datasets"].values()
        ),
        "correlation_stress": {
            "duplicate_count": correlation["protocol"]["duplicate_count"],
            "max_ate_ratio_difference_from_single_copy": correlation_max_difference,
            "interpretation": (
                "a one-factor cluster budget makes ten identical factors "
                "numerically equivalent to one factor"
            ),
        },
        "fixed_legacy_terminal_set": {
            "legacy_terminal_ate_m": old_c1["terminal_ate_m"],
            "best_profile": terminal_best_name,
            "best_reweighted_ate_m": fixed_ntu["ate_m"],
            "odometry_ate_m": old_c1["odometry_ate_m"],
        },
        "end_to_end_ntu": {
            "legacy_terminal_ate_m": old_c1["terminal_ate_m"],
            "calibrated_terminal_ate_m": new_c1["terminal_ate_m"],
            "odometry_ate_m": new_c1["odometry_ate_m"],
            "legacy_active_constraints": old_c1["active_constraints"],
            "calibrated_active_constraints": new_c1["active_constraints"],
            "calibrated_improvement_over_legacy_m": (
                old_c1["terminal_ate_m"] - new_c1["terminal_ate_m"]
            ),
            "calibrated_degradation_vs_odometry_m": (
                new_c1["terminal_ate_m"] - new_c1["odometry_ate_m"]
            ),
        },
        "decision": {
            "engineering_default": "promote calibrated profile; retain --pgo-profile legacy",
            "paper_novelty": False,
            "reason": (
                "weight calibration and correlation budgeting reduce sensitivity, "
                "but GNC rejects none of the frozen candidates and the end-to-end "
                "system still retains truth-incorrect map loops"
            ),
        },
        "source_manifests": {
            str(path.resolve()): sha256(path)
            for path in (
                args.scale_ablation, args.mode_ablation,
                args.correlation_stress, args.terminal_ablation,
                args.legacy_c2_baselines,
            )
        },
        "code_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                HERE.parents[1] / "gui/manual_loop_closure/python_optimizer/loop_weighting.py",
                HERE.parents[1] / "gui/manual_loop_closure/python_optimizer/optimizer.py",
                HERE / "auto_repair_headless.py",
                Path(__file__).resolve(),
            )
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "loop_weighting_decision.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Loop weighting decision", "",
        "Status: exploratory development result; not held out.", "",
        "## Recommended engineering profile", "",
        "- GNC-TLS for added loops; original graph factors are known inliers.",
        "- Translation sigma 1.2649 m; rotation sigma 0.5 deg.",
        "- Global loop-information scale 0.1.",
        "- 30-keyframe endpoint clustering; at most one effective factor per cluster.",
        "", "## Candidate-pool ATE", "",
        "| Dataset | Odometry | Legacy GNC | Calibrated | Change vs odometry |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, item in candidate_pool.items():
        lines.append(
            f"| {tag} | {item['odometry_ate_m']:.4f} | "
            f"{item['legacy_gnc_ate_m']:.4f} | {item['calibrated_ate_m']:.4f} | "
            f"{item['calibrated_vs_odometry_percent']:+.2f}% |"
        )
    lines.extend([
        "", "## What the experiment establishes", "",
        f"- Fixed old terminal set: {old_c1['terminal_ate_m']:.4f} m -> "
        f"{fixed_ntu['ate_m']:.4f} m after reweighting.",
        f"- Full NTU rerun: {old_c1['terminal_ate_m']:.4f} m -> "
        f"{new_c1['terminal_ate_m']:.4f} m, but odometry remains "
        f"{new_c1['odometry_ate_m']:.4f} m.",
        f"- Ten-copy correlation stress differs from the single-copy profile by at most "
        f"{correlation_max_difference:.3e} in ATE ratio.",
        "- Every frozen GNC candidate retained weight 1.0; robust optimization did not "
        "identify the truth-incorrect constraints.",
        "", "## Decision", "",
        "Promote this as an engineering default, not a paper contribution. It reduces "
        "damage and makes duplicate evidence invariant, but cannot make an incorrect "
        "loop measurement correct or replace independent candidate validation.",
    ])
    markdown_path = output_dir / "LOOP_WEIGHTING_DECISION.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (json_path, markdown_path):
        (path.parent / f"{path.name}.sha256").write_text(
            f"{sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
