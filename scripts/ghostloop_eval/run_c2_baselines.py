#!/usr/bin/env python3
"""Run Huber-PGO, GNC-TLS, and odometry-budget C2 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUI = HERE.parents[1] / "gui"
sys.path.insert(0, str(GUI))

from manual_loop_closure import (  # noqa: E402
    OFFICE_DEFAULT_VARIANCE_R_RAD2,
    OFFICE_DEFAULT_VARIANCE_T,
)
from experiment_io import CONSTRAINT_HEADER  # noqa: E402
from evaluate_icra2027_rebuild import (  # noqa: E402
    evaluate,
    sha256_file,
    trajectory_metric,
)


CLI = GUI / "manual_loop_closure/python_optimizer/cli.py"
SIGMA_T = math.sqrt(OFFICE_DEFAULT_VARIANCE_T[0])
SIGMA_R = math.sqrt(OFFICE_DEFAULT_VARIANCE_R_RAD2[0])


def constraint_row(row: dict) -> list[str]:
    return [
        "1", str(row["source_id"]), str(row["target_id"]),
        *(f"{float(row[name]):.12f}" for name in (
            "measurement_tx", "measurement_ty", "measurement_tz",
            "measurement_qx", "measurement_qy", "measurement_qz",
            "measurement_qw",
        )),
        *(f"{SIGMA_T:.6f}" for _ in range(3)),
        *(f"{SIGMA_R:.6f}" for _ in range(3)),
    ]


def run_solver(
    session: Path,
    baseline_session: Path,
    name: str,
    mode: str,
    rows: list[dict],
) -> dict:
    output = baseline_session / name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite C2 baseline: {output}")
    output.mkdir(parents=True)
    constraints = output / "candidate_constraints.csv"
    with constraints.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CONSTRAINT_HEADER)
        writer.writerows(constraint_row(row) for row in rows)
    command = [
        sys.executable, "-u", str(CLI),
        "--session-root", str(session),
        "--g2o", str(session / "pose_graph.g2o"),
        "--tum", str(session / "optimized_poses_tum.txt"),
        "--keyframe-dir", str(session / "key_point_frame"),
        "--constraints-csv", str(constraints),
        "--output-dir", str(output),
        "--optimize-mode", mode,
        "--skip-map-build",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (output / "solver.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"{session.name}/{name} failed")
    report = json.loads((output / "manual_loop_report.json").read_text())
    weights = report.get("robust_weights")
    candidate_weights = (
        None if weights is None else ([] if not rows else weights[-len(rows):])
    )
    return {
        "output_dir": str(output.resolve()),
        "trajectory": str((output / "optimized_poses_tum.txt").resolve()),
        "constraints_sha256": sha256_file(constraints),
        "candidate_count": len(rows),
        "candidate_proposal_ids": [row["proposal_id"] for row in rows],
        "candidate_weights": candidate_weights,
        "solver_report_sha256": sha256_file(output / "manual_loop_report.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        help=(
            "Independent directory for solver outputs and manifest. Defaults "
            "to the rebuild root only for backward compatibility."
        ),
    )
    args = parser.parse_args()
    root = args.rebuild_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve() if args.output_root else root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    source = evaluate(root)
    by_dataset: dict[str, list[dict]] = {}
    for row in source["proposal_rows"]:
        if row["source"] == "stage0_descriptor":
            by_dataset.setdefault(row["dataset"], []).append(row)
    output = {}
    for tag, candidates in sorted(by_dataset.items()):
        session = root / tag
        manifest = json.loads((session / "input_manifest.json").read_text())
        gicp = [row for row in candidates if row["gicp_gate_passed"]]
        budget = [
            row for row in gicp
            if row["odometry_budget_occupancy"] is not None
            and row["odometry_budget_occupancy"] <= 1.0
        ]
        full = [row for row in gicp if row["active_final"]]
        baseline_session = output_root / tag
        arms = {
            "gicp_huber_pgo": run_solver(
                session, baseline_session, "gicp_huber_pgo", "lm_huber", gicp
            ),
            "gnc_tls": run_solver(
                session, baseline_session, "gnc_tls", "gnc_tls", gicp
            ),
            "odometry_budget": run_solver(
                session, baseline_session, "odometry_budget", "isam2", budget
            ),
        }
        summary = json.loads((session / "auto_repair_summary.json").read_text())
        terminal = Path(summary["terminal_output_dir"]) / "optimized_poses_tum.txt"
        arms["full_audit"] = {
            "output_dir": summary["terminal_output_dir"],
            "trajectory": str(terminal.resolve()),
            "candidate_count": len(full),
            "candidate_proposal_ids": [row["proposal_id"] for row in full],
            "candidate_weights": None,
        }
        for arm in arms.values():
            arm["ate_m"] = trajectory_metric(
                tag, Path(arm["trajectory"]), manifest
            )
        output[tag] = arms
    manifest_path = output_root / "c2_baseline_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (output_root / "c2_baseline_manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        tag: {name: {"n": arm["candidate_count"], "ate_m": arm["ate_m"]}
              for name, arm in arms.items()}
        for tag, arms in output.items()
    }, indent=2))


if __name__ == "__main__":
    main()
