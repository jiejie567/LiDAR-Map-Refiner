#!/usr/bin/env python3
"""Re-solve frozen terminal constraints with source-specific information scales."""
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


CLI = GUI / "manual_loop_closure/python_optimizer/cli.py"


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _pair(row: dict) -> tuple[int, int]:
    return int(row["source_id"]), int(row["target_id"])


def _write_constraints(
    source_path: Path, output_path: Path, source_by_pair: dict[tuple[int, int], str],
    stage0_scale: float, map_scale: float,
) -> list[dict]:
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    for field in ("sigma_rotation_unit", "confidence", "factor_information_scale"):
        if field not in header:
            header.append(field)
    audit = []
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for row in rows:
            pair = _pair(row)
            source = source_by_pair.get(pair)
            if source is None:
                source = source_by_pair.get((pair[1], pair[0]))
            if source is None:
                raise KeyError(f"terminal constraint {pair} has no proposal provenance")
            scale = map_scale if source == "map_inconsistency" else stage0_scale
            row["sigma_rotation_unit"] = row.get("sigma_rotation_unit") or "deg"
            row["confidence"] = row.get("confidence") or "1.0"
            row["factor_information_scale"] = f"{scale:.12g}"
            writer.writerow(row)
            audit.append({"pair": pair, "source": source, "factor_scale": scale})
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--stage0-scale", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--map-scale", type=float, nargs="+", default=[0.001, 0.01, 0.1, 0.3, 1.0])
    parser.add_argument("--mode", choices=["lm", "isam2", "lm_huber", "gnc_tls"], default="gnc_tls")
    args = parser.parse_args()
    root = args.rebuild_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence = evaluate(root)
    active_by_dataset: dict[str, dict[tuple[int, int], str]] = {}
    for row in evidence["proposal_rows"]:
        if row["active_final"]:
            active_by_dataset.setdefault(row["dataset"], {})[_pair(row)] = row["source"]

    datasets = {}
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = session / "auto_repair_summary.json"
        manifest_path = session / "input_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        terminal = Path(summary["terminal_output_dir"])
        source_csv = terminal / "manual_loop_constraints.csv"
        if not source_csv.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        datasets[session.name] = {
            "session": session,
            "source_csv": source_csv,
            "manifest": manifest,
            "odometry_ate_m": trajectory_metric(
                session.name, session / "optimized_poses_tum.txt", manifest
            ),
        }

    profiles = {}
    for stage0_scale, map_scale in itertools.product(args.stage0_scale, args.map_scale):
        name = f"{args.mode}_stage0_{_tag(stage0_scale)}_map_{_tag(map_scale)}"
        result = {"stage0_scale": stage0_scale, "map_scale": map_scale, "datasets": {}}
        for tag, item in datasets.items():
            with item["source_csv"].open(newline="", encoding="utf-8") as stream:
                count = sum(1 for _ in csv.DictReader(stream))
            if count == 0:
                result["datasets"][tag] = {
                    "constraint_count": 0,
                    "ate_m": item["odometry_ate_m"],
                    "ate_ratio": 1.0,
                }
                continue
            output = output_root / "runs" / name / tag
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            output.mkdir(parents=True)
            constraints = output / "manual_loop_constraints.csv"
            audit = _write_constraints(
                item["source_csv"], constraints,
                active_by_dataset.get(tag, {}), stage0_scale, map_scale,
            )
            command = [
                sys.executable, "-u", str(CLI),
                "--session-root", str(item["session"]),
                "--g2o", str(item["session"] / "pose_graph.g2o"),
                "--tum", str(item["session"] / "optimized_poses_tum.txt"),
                "--keyframe-dir", str(item["session"] / "key_point_frame"),
                "--constraints-csv", str(constraints),
                "--output-dir", str(output),
                "--optimize-mode", args.mode,
                "--skip-map-build", "--skip-graph-plot",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            (output / "solver.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            if completed.returncode:
                raise RuntimeError(f"solver failed: {output}")
            trajectory = output / "optimized_poses_tum.txt"
            ate = trajectory_metric(tag, trajectory, item["manifest"])
            report = json.loads((output / "manual_loop_report.json").read_text())
            weights = report.get("robust_weights")
            result["datasets"][tag] = {
                "constraint_count": count,
                "constraint_audit": audit,
                "constraints_sha256": sha256_file(constraints),
                "ate_m": ate,
                "ate_ratio": ate / item["odometry_ate_m"],
                "candidate_weights": None if weights is None else weights[-count:],
                "output_dir": str(output.resolve()),
            }
        ratios = [x["ate_ratio"] for x in result["datasets"].values()]
        result["geometric_mean_ate_ratio"] = math.exp(
            sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios)
        )
        result["worst_ate_ratio"] = max(ratios)
        profiles[name] = result
        print(json.dumps({name: {"mean": result["geometric_mean_ate_ratio"], "worst": result["worst_ate_ratio"]}}), flush=True)

    safe = [(name, x) for name, x in profiles.items() if x["worst_ate_ratio"] <= 1.02]
    best_name, _ = min(
        safe or profiles.items(),
        key=lambda pair: (pair[1]["geometric_mean_ate_ratio"], pair[1]["worst_ate_ratio"]),
    )
    payload = {
        "protocol": "exploratory source-specific terminal-factor scaling",
        "rebuild_root": str(root),
        "best_profile": best_name,
        "profiles": profiles,
    }
    path = output_root / "terminal_constraint_weighting_ablation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_root / "terminal_constraint_weighting_ablation.json.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_profile": best_name, "result": profiles[best_name]}, indent=2))


if __name__ == "__main__":
    main()
