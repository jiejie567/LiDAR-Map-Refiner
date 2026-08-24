#!/usr/bin/env python3
"""Grid-search calibrated, correlation-aware loop fusion on frozen proposals.

This is an exploratory engineering ablation, not a held-out paper result.  It
keeps every pose-graph measurement and every proposed relative transform fixed;
only loop-factor uncertainty, correlation budget, and robust solver vary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUI = HERE.parents[1] / "gui"
sys.path.insert(0, str(GUI))

from evaluate_icra2027_rebuild import evaluate, sha256_file, trajectory_metric  # noqa: E402
from experiment_io import CONSTRAINT_HEADER  # noqa: E402


CLI = GUI / "manual_loop_closure/python_optimizer/cli.py"
EXTRA_HEADER = ["sigma_rotation_unit", "confidence"]
INFORMATION_HEADER = "information_upper_json"


def _float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _profile_name(
    mode: str, sigma_t: float, sigma_r_deg: float, information_scale: float,
    window: int, budget: float,
) -> str:
    return (
        f"{mode}_t{_float_tag(sigma_t)}_r{_float_tag(sigma_r_deg)}deg"
        f"_s{_float_tag(information_scale)}_w{window}_b{_float_tag(budget)}"
    )


def _constraint_row(
    row: dict, sigma_t: float, sigma_r_deg: float, *, include_information: bool = False,
) -> list[str]:
    values = [
        "1", str(row["source_id"]), str(row["target_id"]),
        *(f"{float(row[name]):.12f}" for name in (
            "measurement_tx", "measurement_ty", "measurement_tz",
            "measurement_qx", "measurement_qy", "measurement_qz",
            "measurement_qw",
        )),
        *(f"{sigma_t:.12g}" for _ in range(3)),
        *(f"{sigma_r_deg:.12g}" for _ in range(3)),
        "deg", "1.0",
    ]
    if include_information:
        values.append(str(row.get(INFORMATION_HEADER) or ""))
    return values


def _run_one(
    *, session: Path, output: Path, rows: list[dict], mode: str,
    sigma_t: float, sigma_r_deg: float, information_scale: float,
    window: int, budget: float, resume: bool,
    correlation_policy: str = "pair",
    cluster_allocation: str = "equal",
) -> dict:
    report_path = output / "manual_loop_report.json"
    trajectory = output / "optimized_poses_tum.txt"
    constraints = output / "candidate_constraints.csv"
    if resume and report_path.is_file() and trajectory.is_file() and constraints.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete output: {output}")
        output.mkdir(parents=True)
        include_information = any(
            bool(row.get(INFORMATION_HEADER)) for row in rows
        )
        with constraints.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            header = [*CONSTRAINT_HEADER, *EXTRA_HEADER]
            if include_information:
                header.append(INFORMATION_HEADER)
            writer.writerow(header)
            writer.writerows(
                _constraint_row(
                    row, sigma_t, sigma_r_deg,
                    include_information=include_information,
                )
                for row in rows
            )
        command = [
            sys.executable, "-u", str(CLI),
            "--session-root", str(session),
            "--g2o", str(session / "pose_graph.g2o"),
            "--tum", str(session / "optimized_poses_tum.txt"),
            "--keyframe-dir", str(session / "key_point_frame"),
            "--constraints-csv", str(constraints),
            "--output-dir", str(output),
            "--optimize-mode", mode,
            "--manual-information-scale", str(information_scale),
            "--loop-correlation-window-keyframes", str(window),
            "--loop-cluster-information-budget", str(budget),
            "--loop-correlation-policy", correlation_policy,
            "--loop-cluster-allocation", cluster_allocation,
            "--skip-map-build", "--skip-graph-plot",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (output / "solver.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"solver failed: {output}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = report.get("robust_weights")
    candidate_weights = None if weights is None else weights[-len(rows):] if rows else []
    return {
        "output_dir": str(output.resolve()),
        "trajectory": str(trajectory.resolve()),
        "constraints_sha256": sha256_file(constraints),
        "candidate_count": len(rows),
        "candidate_proposal_ids": [row["proposal_id"] for row in rows],
        "candidate_weights": candidate_weights,
        "solver_report_sha256": sha256_file(report_path),
        "manual_factor_weighting": report.get("manual_factor_weighting"),
    }


def _score(profile: dict, odometry: dict[str, float], arm: str) -> dict:
    ratios = {
        tag: float(result[arm]["ate_m"]) / float(odometry[tag])
        for tag, result in profile["datasets"].items()
    }
    values = list(ratios.values())
    return {
        "ate_ratio_by_dataset": ratios,
        "geometric_mean_ate_ratio": math.exp(
            sum(math.log(max(value, 1e-12)) for value in values) / len(values)
        ),
        "worst_ate_ratio": max(values),
        "improved_dataset_count": sum(value < 1.0 for value in values),
        "safe_within_2_percent": max(values) <= 1.02,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sigma-t", type=float, nargs="+", default=[0.2, 0.5, 1.0, math.sqrt(1.6)])
    parser.add_argument("--sigma-r-deg", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, math.degrees(math.sqrt(0.1))])
    parser.add_argument("--information-scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--correlation-window", type=int, nargs="+", default=[0])
    parser.add_argument("--cluster-budget", type=float, nargs="+", default=[0.0])
    parser.add_argument("--mode", nargs="+", default=["gnc_tls"], choices=["lm", "isam2", "lm_huber", "gnc_tls"])
    parser.add_argument(
        "--duplicate-count", type=int, default=1,
        help="Controlled correlation stress test: repeat every factor this many times.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.duplicate_count <= 0:
        raise SystemExit("--duplicate-count must be positive")

    root = args.rebuild_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = evaluate(root)
    by_dataset: dict[str, list[dict]] = {}
    for row in source["proposal_rows"]:
        if row["source"] == "stage0_descriptor" and row["gicp_gate_passed"]:
            by_dataset.setdefault(row["dataset"], []).append(row)

    odometry: dict[str, float] = {}
    manifests: dict[str, dict] = {}
    for tag in sorted(by_dataset):
        manifest = json.loads((root / tag / "input_manifest.json").read_text())
        manifests[tag] = manifest
        odometry[tag] = trajectory_metric(
            tag, root / tag / "optimized_poses_tum.txt", manifest
        )

    profiles = {}
    combinations = itertools.product(
        args.mode, args.sigma_t, args.sigma_r_deg, args.information_scale,
        args.correlation_window, args.cluster_budget,
    )
    for mode, sigma_t, sigma_r_deg, scale, window, budget in combinations:
        name = _profile_name(mode, sigma_t, sigma_r_deg, scale, window, budget)
        profile = {
            "config": {
                "mode": mode, "sigma_t_m": sigma_t,
                "sigma_r_deg": sigma_r_deg, "information_scale": scale,
                "correlation_window_keyframes": window,
                "cluster_information_budget": budget,
            },
            "datasets": {},
        }
        for tag, unique_rows in sorted(by_dataset.items()):
            all_rows = [
                row for row in unique_rows for _ in range(args.duplicate_count)
            ]
            session = root / tag
            truth_rows = [
                row for row in all_rows
                if row.get("gt_available")
                and row.get("relative_translation_error_m") is not None
                and row.get("relative_rotation_error_deg") is not None
                and float(row["relative_translation_error_m"]) <= 1.0
                and float(row["relative_rotation_error_deg"]) <= 5.0
            ]
            arms = {}
            for arm, rows in (("all", all_rows), ("truth_only", truth_rows)):
                if rows:
                    result = _run_one(
                        session=session,
                        output=output_root / "runs" / name / arm / tag,
                        rows=rows, mode=mode, sigma_t=sigma_t,
                        sigma_r_deg=sigma_r_deg, information_scale=scale,
                        window=window, budget=budget, resume=args.resume,
                    )
                    result["ate_m"] = trajectory_metric(
                        tag, Path(result["trajectory"]), manifests[tag]
                    )
                else:
                    result = {
                        "output_dir": None,
                        "trajectory": str((session / "optimized_poses_tum.txt").resolve()),
                        "candidate_count": 0,
                        "candidate_proposal_ids": [],
                        "candidate_weights": [],
                        "ate_m": odometry[tag],
                    }
                arms[arm] = result
            profile["datasets"][tag] = arms
        profile["all_score"] = _score(profile, odometry, "all")
        profile["truth_only_score"] = _score(profile, odometry, "truth_only")
        profiles[name] = profile
        print(json.dumps({name: profile["all_score"]}, sort_keys=True), flush=True)

    safe = [
        (name, item) for name, item in profiles.items()
        if item["all_score"]["safe_within_2_percent"]
    ]
    ranking_pool = safe or list(profiles.items())
    best_name, _best = min(
        ranking_pool,
        key=lambda pair: (
            pair[1]["all_score"]["geometric_mean_ate_ratio"],
            pair[1]["all_score"]["worst_ate_ratio"],
        ),
    )
    manifest = {
        "protocol": {
            "status": "exploratory_development_ablation",
            "selection_rule": (
                "lowest geometric-mean all-candidate ATE ratio among profiles "
                "with no dataset worse than 2%; if none, lowest mean ratio"
            ),
            "rebuild_root": str(root),
            "source_evidence_sha256": source.get("source_data_sha256"),
            "duplicate_count": int(args.duplicate_count),
        },
        "odometry_ate_m": odometry,
        "best_profile": best_name,
        "profiles": profiles,
    }
    manifest_path = output_root / "loop_weighting_ablation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_root / "loop_weighting_ablation.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_profile": best_name, "score": profiles[best_name]["all_score"]}, indent=2))


if __name__ == "__main__":
    main()
