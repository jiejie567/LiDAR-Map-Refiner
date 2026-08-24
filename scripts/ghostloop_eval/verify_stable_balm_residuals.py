#!/usr/bin/env python3
"""Verify residual-ghost loop proposals after repeated BALM convergence attempts.

This is an experiment-only one-round probe. It holds the input trajectory and
With the Initial Loop Search constraint checkpoint fixed, takes the ten reported residual regions,
and applies the same indoor registration cascade, gates, signed separation
prior, descriptor-yaw prior, near-duplicate suppression, and odometry budget as
``auto_repair_headless.py``. It writes an augmented constraint CSV but does not
run PGO or BALM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GUI_DIR = SCRIPT_DIR.parents[1] / "gui"
sys.path.insert(0, str(GUI_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from manual_loop_closure import (  # noqa: E402
    OFFICE_DEFAULT_VARIANCE_R_RAD2,
    OFFICE_DEFAULT_VARIANCE_T,
    RegistrationConfig,
    RegistrationWorkspace,
    load_tum_trajectory,
    matrix_to_quat_xyzw,
)
from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
    BalmParams,
    prepare_local_cloud,
)
from manual_loop_closure.repair_presets import (  # noqa: E402
    NEAR_DUPLICATE_KEYFRAMES,
    REPAIR_POLICY,
    repair_environment_preset,
    repair_gate,
)
from manual_loop_closure.scan_context_io import (  # noqa: E402
    _gravity_canonical_rotation,
    load_scan_context_config,
    load_scan_context_gravity,
)
from manual_loop_closure.seed_loops import (  # noqa: E402
    build_descriptor_and_prepared_stack,
    descriptor_env_setup,
    pair_yaw_peaks,
)
from experiment_io import CONSTRAINT_HEADER, load_initial_constraints_csv  # noqa: E402


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--session", type=Path, required=True)
    out.add_argument("--tum", type=Path, required=True)
    out.add_argument("--balm-report", type=Path, required=True)
    out.add_argument("--initial-constraints-csv", type=Path, required=True)
    out.add_argument("--output-dir", type=Path, required=True)
    out.add_argument("--environment", choices=("indoor", "outdoor"), default="indoor")
    out.add_argument("--max-proposals", type=int, default=10)
    return out


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = args.session / "key_point_frame"
    trajectory = load_tum_trajectory(args.tum)
    workspace = RegistrationWorkspace(keyframes, trajectory)
    config, channel_weights = descriptor_env_setup(
        load_scan_context_config(args.session), args.environment
    )
    preset = repair_environment_preset(args.environment)
    gate = repair_gate(preset["voxel"])
    min_fitness = float(gate["min_fitness"])
    max_rmse = float(gate["max_rmse"])
    reg_config = RegistrationConfig(
        max_correspondence_distance=preset["max_corr"],
        voxel_size=preset["voxel"],
        target_map_voxel_size=preset["target_map_voxel"],
        target_neighbors=preset["target_neighbors"],
    )
    wide_config = replace(
        reg_config,
        max_correspondence_distance=preset["wide_max_corr"],
        max_iterations=100,
    )
    submap_config = replace(wide_config, source_window=10)
    anneal_config = wide_config
    balm_params = BalmParams(
        root_voxel_size=preset["balm_voxel"],
        downsample_leaf=preset["balm_downsample"],
        max_range=preset["balm_max_range"],
        max_iterations=preset["balm_iterations"],
        double_sided_enable=preset["balm_double_sided"],
    )
    diagnosis_params = replace(
        balm_params,
        max_range=min(balm_params.max_range, preset["observation_range"]),
    )

    gravity = None
    if config.gravity_canonicalization_enable:
        gravity = load_scan_context_gravity(args.session, trajectory)
    descriptors, masks, _ = build_descriptor_and_prepared_stack(
        workspace.load_local_points_uncached,
        lambda points: prepare_local_cloud(points, diagnosis_params),
        trajectory.size,
        config,
        gravity,
        log_fn=lambda message: print(message, flush=True),
    )

    transforms = trajectory.transforms_world_sensor
    cumulative_distance = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(trajectory.positions_xyz, axis=0), axis=1))]
    )
    budget_base = REPAIR_POLICY[
        "budget_base_outdoor_m" if args.environment == "outdoor" else "budget_base_indoor_m"
    ]
    budget_rate = float(REPAIR_POLICY["budget_rate"])
    near_duplicate = int(NEAR_DUPLICATE_KEYFRAMES)

    def desired_rotation(source_id: int, target_id: int, yaw_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        rz = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if gravity is None:
            return rz
        return (
            _gravity_canonical_rotation(gravity[target_id]).T
            @ rz
            @ _gravity_canonical_rotation(gravity[source_id])
        )

    def attempt(source_id: int, target_id: int, delta: np.ndarray, cfg: RegistrationConfig):
        try:
            result = workspace.run_gicp(
                source_id=source_id,
                target_id=target_id,
                delta_transform_local=delta,
                config=cfg,
            )
        except Exception:
            return None
        return result

    def pass_gate(result) -> bool:
        return result.fitness >= min_fitness and result.inlier_rmse <= max_rmse

    def best_passing(source_id: int, target_id: int, deltas: list[np.ndarray]):
        all_attempts: list[dict] = []
        for cfg, tier in (
            (reg_config, "near"),
            (wide_config, "wide"),
            (submap_config, "submap"),
        ):
            candidates = []
            for seed_index, delta in enumerate(deltas):
                result = attempt(source_id, target_id, delta, cfg)
                if result is None:
                    all_attempts.append({"tier": tier, "seed": seed_index, "status": "error"})
                    continue
                all_attempts.append(
                    {
                        "tier": tier,
                        "seed": seed_index,
                        "fitness": float(result.fitness),
                        "inlier_rmse_m": float(result.inlier_rmse),
                        "passed_pair_gate": pass_gate(result),
                    }
                )
                if pass_gate(result):
                    candidates.append((float(result.inlier_rmse), seed_index, result))
            if candidates:
                _, seed_index, result = min(candidates)
                return result, tier, seed_index, all_attempts

        candidates = []
        for seed_index, delta in enumerate(deltas):
            warm = attempt(source_id, target_id, delta, anneal_config)
            if warm is None:
                all_attempts.append({"tier": "annealed", "seed": seed_index, "status": "warm_error"})
                continue
            next_delta = (
                np.linalg.inv(transforms[source_id])
                @ transforms[target_id]
                @ warm.transform_target_source_final
            )
            result = attempt(source_id, target_id, next_delta, reg_config)
            if result is None:
                all_attempts.append({"tier": "annealed", "seed": seed_index, "status": "refine_error"})
                continue
            all_attempts.append(
                {
                    "tier": "annealed",
                    "seed": seed_index,
                    "fitness": float(result.fitness),
                    "inlier_rmse_m": float(result.inlier_rmse),
                    "passed_pair_gate": pass_gate(result),
                }
            )
            if pass_gate(result):
                candidates.append((float(result.inlier_rmse), seed_index, result))
        if candidates:
            _, seed_index, result = min(candidates)
            return result, "annealed", seed_index, all_attempts
        return None, None, None, all_attempts

    initial_rows = load_initial_constraints_csv(args.initial_constraints_csv, trajectory.size)
    output_rows = [list(row) for row in initial_rows]
    report = json.loads(args.balm_report.read_text(encoding="utf-8"))
    regions = (report.get("ghost_regions") or [])[: max(0, args.max_proposals)]
    results: list[dict] = []
    sigma_t = f"{math.sqrt(OFFICE_DEFAULT_VARIANCE_T[0]):.6f}"
    sigma_r = f"{math.sqrt(OFFICE_DEFAULT_VARIANCE_R_RAD2[0]):.6f}"
    seen_pairs = {(str(row[1]), str(row[2])) for row in initial_rows}

    for rank, region in enumerate(regions, start=1):
        low_frame, high_frame = (int(value) for value in region["suggested_pair"])
        source_id, target_id = max(low_frame, high_frame), min(low_frame, high_frame)
        source_text, target_text = str(source_id), str(target_id)
        item = {
            "rank": rank,
            "center_xyz": region.get("center_xyz"),
            "reported_separation_m": float(region.get("separation_m", 0.0)),
            "suggested_pair": [low_frame, high_frame],
            "ordered_pair": [source_id, target_id],
        }
        if (source_text, target_text) in seen_pairs or source_id == target_id:
            item["status"] = "duplicate"
            results.append(item)
            continue
        if any(
            abs(source_id - int(row[1])) <= near_duplicate
            and abs(target_id - int(row[2])) <= near_duplicate
            for row in output_rows
        ):
            item["status"] = "near_duplicate"
            results.append(item)
            continue
        seen_pairs.add((source_text, target_text))

        peaks = pair_yaw_peaks(
            descriptors,
            masks,
            source_id,
            target_id,
            channel_weights,
            config.num_rings,
            min_joint_rings=config.min_joint_rings,
            retrieval_height_offset=config.retrieval_height_offset,
            sector_support_exponent=config.sector_support_exponent,
        )
        relative = np.linalg.inv(transforms[target_id]) @ transforms[source_id]
        deltas = [np.eye(4)]
        distance, yaw_deg = peaks[0]
        desired = desired_rotation(source_id, target_id, yaw_deg)
        if abs(yaw_deg) > 3.0:
            sc_delta = np.eye(4)
            sc_delta[:3, :3] = relative[:3, :3].T @ desired
            deltas.append(sc_delta)
        normal = np.asarray(region.get("normal", [0.0, 0.0, 0.0]), dtype=float)
        separation = float(region.get("separation_m", 0.0))
        separation_sign = 1.0 if source_id == low_frame else -1.0
        if np.linalg.norm(normal) > 0.5 and separation > 0.0:
            local_direction = transforms[source_id][:3, :3].T @ normal
            for base in list(deltas)[:2]:
                delta = base.copy()
                delta[:3, 3] += separation_sign * separation * local_direction
                deltas.append(delta)

        accepted, tier, seed_index, attempts = best_passing(
            source_id, target_id, deltas
        )
        item.update(
            {
                "descriptor_distance": float(distance),
                "descriptor_yaw_deg": float(yaw_deg),
                "seed_count": len(deltas),
                "attempts": attempts,
            }
        )
        if accepted is None:
            item["status"] = "pair_gate_rejected"
            results.append(item)
            continue

        odometry_relative = np.linalg.inv(transforms[target_id]) @ transforms[source_id]
        correction = float(
            np.linalg.norm(
                (
                    np.linalg.inv(odometry_relative)
                    @ accepted.transform_target_source_final
                )[:3, 3]
            )
        )
        chain_distance = float(abs(cumulative_distance[source_id] - cumulative_distance[target_id]))
        budget = float(budget_base + budget_rate * chain_distance)
        item.update(
            {
                "winning_tier": tier,
                "winning_seed": int(seed_index),
                "fitness": float(accepted.fitness),
                "inlier_rmse_m": float(accepted.inlier_rmse),
                "odometry_correction_m": correction,
                "odometry_chain_m": chain_distance,
                "odometry_budget_m": budget,
            }
        )
        if correction > budget:
            item["status"] = "odometry_budget_rejected"
            results.append(item)
            continue

        transform = accepted.transform_target_source_final
        quaternion = matrix_to_quat_xyzw(transform)
        output_rows.append(
            [
                "1",
                source_text,
                target_text,
                *(f"{value:.12f}" for value in transform[:3, 3]),
                *(f"{value:.12f}" for value in quaternion),
                sigma_t,
                sigma_t,
                sigma_t,
                sigma_r,
                sigma_r,
                sigma_r,
            ]
        )
        item["status"] = "accepted"
        results.append(item)
        print(
            f"rank {rank}: {target_id}->{source_id} ACCEPT {tier} "
            f"fit={accepted.fitness:.3f} rmse={accepted.inlier_rmse:.3f}",
            flush=True,
        )

    constraints_out = args.output_dir / "stable_residual_constraints.csv"
    with constraints_out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CONSTRAINT_HEADER)
        writer.writerows(output_rows)

    summary = {
        "input_tum": str(args.tum.resolve()),
        "input_balm_report": str(args.balm_report.resolve()),
        "initial_constraint_count": len(initial_rows),
        "proposal_count": len(regions),
        "accepted_count": sum(item["status"] == "accepted" for item in results),
        "final_constraint_count": len(output_rows),
        "pair_gate": {"min_fitness": min_fitness, "max_rmse_m": max_rmse},
        "registration": {
            "near": asdict(reg_config),
            "wide": asdict(wide_config),
            "submap": asdict(submap_config),
        },
        "results": results,
    }
    (args.output_dir / "stable_residual_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "initial_constraint_count", "proposal_count", "accepted_count", "final_constraint_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
