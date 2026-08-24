#!/usr/bin/env python3
"""Measure ATE as truth-confirmed place loops are added one by one.

Place correctness is intentionally separate from relative-pose accuracy: a
candidate belongs to the same physical neighbourhood when its two ground-truth
endpoints are within ``--place-radius-m``, even if ICP returns a biased SE(3)
measurement.  This experiment asks whether PGO weighting handles that bias.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_icra2027_rebuild import (  # noqa: E402
    FRAME_CONFIG,
    evaluate,
    interpolated_truth_in_keyframe_frame,
    trajectory_metric,
    tum_transforms,
)
from run_loop_weighting_ablation import _profile_name, _run_one  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--place-radius-m", type=float, default=2.0)
    parser.add_argument(
        "--max-relative-translation-error-m", type=float,
        help=(
            "Optionally require the frozen loop measurement itself to be "
            "truth-accurate, in addition to the place-association test."
        ),
    )
    parser.add_argument(
        "--max-relative-rotation-error-deg", type=float,
        help=(
            "Optional SO(3) geodesic error limit for the frozen loop "
            "measurement."
        ),
    )
    parser.add_argument("--sigma-t", type=float, nargs="+", required=True)
    parser.add_argument("--sigma-r-deg", type=float, nargs="+", required=True)
    parser.add_argument("--information-scale", type=float, nargs="+", required=True)
    parser.add_argument("--mode", nargs="+", default=["gnc_tls"])
    parser.add_argument("--correlation-window", type=int, default=0)
    parser.add_argument("--cluster-budget", type=float, default=0.0)
    parser.add_argument(
        "--correlation-policy", choices=("pair", "shared_endpoint"),
        default="pair",
    )
    parser.add_argument(
        "--cluster-allocation", choices=("equal", "representative"),
        default="equal",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--measurement-source", choices=("frozen", "truth"),
        default="frozen",
        help="truth is an oracle diagnostic for separating association from ICP bias.",
    )
    args = parser.parse_args()
    root = args.rebuild_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = evaluate(root)
    candidates: dict[str, list[dict]] = {}
    for row in source["proposal_rows"]:
        distance = row.get("gt_endpoint_distance_m")
        translation_error = row.get("relative_translation_error_m")
        rotation_error = row.get("relative_rotation_error_deg")
        measurement_within_limits = (
            (
                args.max_relative_translation_error_m is None
                or (
                    translation_error is not None
                    and float(translation_error)
                    <= args.max_relative_translation_error_m
                )
            )
            and (
                args.max_relative_rotation_error_deg is None
                or (
                    rotation_error is not None
                    and float(rotation_error)
                    <= args.max_relative_rotation_error_deg
                )
            )
        )
        if (
            row["source"] == "stage0_descriptor"
            and row["gicp_gate_passed"]
            and row.get("gt_available")
            and distance is not None
            and float(distance) <= args.place_radius_m
            and measurement_within_limits
        ):
            candidates.setdefault(row["dataset"], []).append(row)
    for rows in candidates.values():
        rows.sort(key=lambda row: (
            float(row["descriptor_distance"]),
            int(row["source_id"]), int(row["target_id"]),
        ))

    dataset_meta = {}
    for tag in candidates:
        session = root / tag
        manifest = json.loads((session / "input_manifest.json").read_text())
        if args.measurement_source == "truth":
            timestamps, _ = tum_transforms(session / "optimized_poses_tum.txt")
            frame_config = json.loads(FRAME_CONFIG.read_text(encoding="utf-8"))
            truth, valid, _ = interpolated_truth_in_keyframe_frame(
                timestamps, Path(manifest["truth"]), tag, frame_config,
                keyframe_frame_role=(manifest.get("keyframe_export") or {}).get(
                    "keyframe_frame_role"
                ),
            )
            for row in candidates[tag]:
                source_id, target_id = int(row["source_id"]), int(row["target_id"])
                if not (valid[source_id] and valid[target_id]):
                    raise RuntimeError(f"truth unavailable for {tag} {source_id}->{target_id}")
                transform = np.linalg.inv(truth[target_id]) @ truth[source_id]
                quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
                for key, value in zip(
                    ("measurement_tx", "measurement_ty", "measurement_tz"),
                    transform[:3, 3],
                ):
                    row[key] = float(value)
                for key, value in zip(
                    ("measurement_qx", "measurement_qy", "measurement_qz", "measurement_qw"),
                    quaternion,
                ):
                    row[key] = float(value)
        odometry = trajectory_metric(
            tag, session / "optimized_poses_tum.txt", manifest
        )
        dataset_meta[tag] = {"session": session, "manifest": manifest, "odometry": odometry}

    profiles = {}
    for mode, sigma_t, sigma_r, scale in itertools.product(
        args.mode, args.sigma_t, args.sigma_r_deg, args.information_scale
    ):
        name = _profile_name(mode, sigma_t, sigma_r, scale, 0, 0.0)
        datasets = {}
        all_steps = []
        all_regrets = []
        for tag, rows in sorted(candidates.items()):
            meta = dataset_meta[tag]
            curve = [float(meta["odometry"])]
            prefix_records = []
            for count in range(1, len(rows) + 1):
                selected = rows[:count]
                result = _run_one(
                    session=meta["session"],
                    output=output_root / "runs" / name / tag / f"prefix_{count:02d}",
                    rows=selected, mode=mode, sigma_t=sigma_t,
                    sigma_r_deg=sigma_r, information_scale=scale,
                    window=args.correlation_window,
                    budget=args.cluster_budget, resume=args.resume,
                    correlation_policy=args.correlation_policy,
                    cluster_allocation=args.cluster_allocation,
                )
                ate = trajectory_metric(
                    tag, Path(result["trajectory"]), meta["manifest"]
                )
                curve.append(float(ate))
                prefix_records.append({
                    "count": count,
                    "added_proposal_id": rows[count - 1]["proposal_id"],
                    "source_id": rows[count - 1]["source_id"],
                    "target_id": rows[count - 1]["target_id"],
                    "gt_endpoint_distance_m": rows[count - 1]["gt_endpoint_distance_m"],
                    "relative_translation_error_m": rows[count - 1]["relative_translation_error_m"],
                    "relative_rotation_error_deg": rows[count - 1]["relative_rotation_error_deg"],
                    "ate_m": ate,
                    "candidate_weights": result.get("candidate_weights"),
                })
            steps = [curve[index] - curve[index - 1] for index in range(1, len(curve))]
            running_best = curve[0]
            regrets = []
            for value in curve[1:]:
                running_best = min(running_best, value)
                regrets.append(value - running_best)
            all_steps.extend(steps)
            all_regrets.extend(regrets)
            datasets[tag] = {
                "candidate_count": len(rows),
                "odometry_ate_m": curve[0],
                "ate_curve_m": curve,
                "prefixes": prefix_records,
                "positive_step_count": sum(step > 1e-6 for step in steps),
                "maximum_positive_step_m": max([0.0, *steps]),
                "maximum_prefix_regret_m": max([0.0, *regrets]),
                "final_ate_ratio": curve[-1] / curve[0],
            }
        profiles[name] = {
            "config": {
                "mode": mode, "sigma_t_m": sigma_t,
                "sigma_r_deg": sigma_r, "information_scale": scale,
                "correlation_window_keyframes": args.correlation_window,
                "cluster_information_budget": args.cluster_budget,
                "correlation_policy": args.correlation_policy,
                "cluster_allocation": args.cluster_allocation,
            },
            "datasets": datasets,
            "positive_step_count": sum(step > 1e-6 for step in all_steps),
            "maximum_positive_step_m": max([0.0, *all_steps]),
            "maximum_prefix_regret_m": max([0.0, *all_regrets]),
            "geometric_mean_final_ate_ratio": math.exp(
                sum(math.log(max(item["final_ate_ratio"], 1e-12)) for item in datasets.values())
                / len(datasets)
            ),
        }
        print(json.dumps({name: {
            key: profiles[name][key] for key in (
                "positive_step_count", "maximum_positive_step_m",
                "maximum_prefix_regret_m", "geometric_mean_final_ate_ratio",
            )
        }}), flush=True)

    # Prefer few regressions, then small regret, but reject the trivial solution
    # of assigning zero information by using final trajectory quality next.
    best_name, _ = min(
        profiles.items(),
        key=lambda pair: (
            pair[1]["positive_step_count"],
            pair[1]["maximum_prefix_regret_m"],
            pair[1]["geometric_mean_final_ate_ratio"],
        ),
    )
    payload = {
        "protocol": {
            "status": "exploratory_development_ablation",
            "definition": "ground-truth endpoint distance <= place_radius_m",
            "place_radius_m": args.place_radius_m,
            "max_relative_translation_error_m": (
                args.max_relative_translation_error_m
            ),
            "max_relative_rotation_error_deg": (
                args.max_relative_rotation_error_deg
            ),
            "order": "ascending descriptor distance",
            "important": "place correctness does not imply accurate SE(3) measurement",
            "measurement_source": args.measurement_source,
        },
        "best_profile": best_name,
        "profiles": profiles,
    }
    path = output_root / "correct_loop_prefix_ablation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_root / "correct_loop_prefix_ablation.json.sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_profile": best_name, "result": profiles[best_name]}, indent=2))


if __name__ == "__main__":
    main()
