#!/usr/bin/env python3
"""Test path-normalized loop information on a frozen prefix experiment.

Each loop is assigned the same scalar Kalman-style gain in translation and
rotation relative to the original adjacent-edge path between its endpoints.
This is an exploratory leverage calibration, not a covariance claim: it asks
whether removing the current channel and keyframe-span imbalance is useful
before touching production behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_correct_loop_prefix_influence import (  # noqa: E402
    chain_channel_variances,
    load_g2o_edges,
)
from evaluate_icra2027_rebuild import trajectory_metric  # noqa: E402
from run_loop_weighting_ablation import _run_one  # noqa: E402


def _float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _upper_json(information: np.ndarray) -> str:
    values = []
    for row in range(6):
        for col in range(row, 6):
            values.append(float(information[row, col]))
    return json.dumps(values, separators=(",", ":"))


def _read_last_constraint(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("enabled") == "1"]
    if not rows:
        raise ValueError(f"no enabled constraints in {path}")
    row = rows[-1]
    return {
        "source_id": int(row["source_id"]),
        "target_id": int(row["target_id"]),
        "measurement_tx": float(row["tx"]),
        "measurement_ty": float(row["ty"]),
        "measurement_tz": float(row["tz"]),
        "measurement_qx": float(row["qx"]),
        "measurement_qy": float(row["qy"]),
        "measurement_qz": float(row["qz"]),
        "measurement_qw": float(row["qw"]),
    }


def _with_target_gain(row: dict, edges, gain: float) -> dict:
    variances = chain_channel_variances(
        int(row["source_id"]), int(row["target_id"]), edges
    )
    if variances is None:
        raise ValueError(
            f"missing contiguous path for {row['target_id']}->{row['source_id']}"
        )
    translation_variance, rotation_variance = variances
    odds = float(gain) / (1.0 - float(gain))
    final_information = np.diag(
        [odds / translation_variance] * 3
        + [odds / rotation_variance] * 3
    )
    return {
        **row,
        "information_upper_json": _upper_json(final_information),
        "path_translation_variance": float(translation_variance),
        "path_rotation_variance": float(rotation_variance),
        "target_channel_gain": float(gain),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_prefix_report", type=Path)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target-channel-gain", type=float, nargs="+", required=True)
    parser.add_argument("--mode", choices=("lm", "gnc_tls"), default="gnc_tls")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for gain in args.target_channel_gain:
        if not 0.0 < gain < 1.0:
            raise SystemExit("--target-channel-gain values must lie in (0, 1)")

    source_path = args.source_prefix_report.expanduser().resolve()
    rebuild_root = args.rebuild_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if len(source["profiles"]) != 1:
        raise ValueError("source prefix report must contain exactly one profile")
    source_profile_name, source_profile = next(iter(source["profiles"].items()))
    source_runs = source_path.parent / "runs" / source_profile_name

    profiles = {}
    for gain in args.target_channel_gain:
        name = f"{args.mode}_path_gain_{_float_tag(gain)}"
        datasets = {}
        all_steps: list[float] = []
        all_regrets: list[float] = []
        for dataset, source_dataset in sorted(source_profile["datasets"].items()):
            session = rebuild_root / dataset
            _, edges = load_g2o_edges(session / "pose_graph.g2o")
            manifest = json.loads((session / "input_manifest.json").read_text())
            candidates = []
            for prefix in source_dataset["prefixes"]:
                count = int(prefix["count"])
                source_constraint = (
                    source_runs / dataset / f"prefix_{count:02d}"
                    / "candidate_constraints.csv"
                )
                candidate = _read_last_constraint(source_constraint)
                candidate["proposal_id"] = prefix["added_proposal_id"]
                candidates.append(_with_target_gain(candidate, edges, gain))

            curve = [float(source_dataset["odometry_ate_m"])]
            prefixes = []
            for count in range(1, len(candidates) + 1):
                selected = candidates[:count]
                result = _run_one(
                    session=session,
                    output=output_root / "runs" / name / dataset / f"prefix_{count:02d}",
                    rows=selected,
                    mode=args.mode,
                    sigma_t=1.0,
                    sigma_r_deg=1.0,
                    information_scale=1.0,
                    window=0,
                    budget=0.0,
                    resume=args.resume,
                )
                ate = trajectory_metric(dataset, Path(result["trajectory"]), manifest)
                if ate is None:
                    raise RuntimeError(f"ATE unavailable for {dataset} prefix {count}")
                curve.append(float(ate))
                prefixes.append({
                    "count": count,
                    "added_proposal_id": candidates[count - 1]["proposal_id"],
                    "source_id": candidates[count - 1]["source_id"],
                    "target_id": candidates[count - 1]["target_id"],
                    "ate_m": float(ate),
                    "candidate_weights": result.get("candidate_weights"),
                    "path_translation_variance": candidates[count - 1][
                        "path_translation_variance"
                    ],
                    "path_rotation_variance": candidates[count - 1][
                        "path_rotation_variance"
                    ],
                })
            steps = [curve[index] - curve[index - 1] for index in range(1, len(curve))]
            running_best = curve[0]
            regrets = []
            for value in curve[1:]:
                running_best = min(running_best, value)
                regrets.append(value - running_best)
            all_steps.extend(steps)
            all_regrets.extend(regrets)
            datasets[dataset] = {
                "candidate_count": len(candidates),
                "odometry_ate_m": curve[0],
                "ate_curve_m": curve,
                "prefixes": prefixes,
                "positive_step_count": sum(step > 1e-6 for step in steps),
                "maximum_positive_step_m": max([0.0, *steps]),
                "maximum_prefix_regret_m": max([0.0, *regrets]),
                "final_ate_ratio": curve[-1] / curve[0],
            }
        geometric_mean = math.exp(sum(
            math.log(max(item["final_ate_ratio"], 1e-12))
            for item in datasets.values()
        ) / len(datasets))
        profiles[name] = {
            "config": {
                "mode": args.mode,
                "target_translation_gain": float(gain),
                "target_rotation_gain": float(gain),
                "path_model": "sum of adjacent original-factor scalar variances",
                "information_scale": 1.0,
            },
            "datasets": datasets,
            "positive_step_count": sum(step > 1e-6 for step in all_steps),
            "maximum_positive_step_m": max([0.0, *all_steps]),
            "maximum_prefix_regret_m": max([0.0, *all_regrets]),
            "geometric_mean_final_ate_ratio": geometric_mean,
        }
        print(json.dumps({name: {
            "geometric_mean_final_ate_ratio": geometric_mean,
            "positive_step_count": profiles[name]["positive_step_count"],
            "maximum_positive_step_m": profiles[name]["maximum_positive_step_m"],
        }}), flush=True)

    best_name = min(
        profiles,
        key=lambda key: (
            profiles[key]["geometric_mean_final_ate_ratio"],
            profiles[key]["maximum_positive_step_m"],
        ),
    )
    output = {
        "protocol": {
            "status": "exploratory_development_ablation",
            "measurement_source": source["protocol"].get("measurement_source"),
            "source_prefix_report": str(source_path),
            "source_prefix_report_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "ground_truth_usage": (
                "measurement source and offline ATE label only; path-normalized "
                "information uses no ground truth"
            ),
        },
        "best_profile": best_name,
        "profiles": profiles,
    }
    output_path = output_root / "path_normalized_loop_ablation.json"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"best_profile": best_name, "result": profiles[best_name]}, indent=2))


if __name__ == "__main__":
    main()
