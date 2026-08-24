#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CURRENT_DIR.parent
GUI_DIR = CURRENT_DIR.parents[1]
# Leaf modules are imported directly (not via the manual_loop_closure package)
# so this backend only needs numpy/scipy, not open3d.
for extra_path in (GUI_DIR, PACKAGE_DIR, CURRENT_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pcd_io import (  # noqa: E402
    list_numbered_pcds,
    load_xyz_points,
    validate_keyframe_numbering,
)
from trajectory_io import load_tum_trajectory  # noqa: E402
from balm import (  # noqa: E402
    BalmParams,
    detect_ghost_regions,
    prepare_local_cloud,
    run_balm_refinement,
)


# ===== BEGIN CHANGE: balm refinement cli =====
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint plane bundle adjustment over an optimized keyframe trajectory."
    )
    parser.add_argument("--tum", required=True, type=Path, help="Input optimized TUM trajectory.")
    parser.add_argument("--keyframe-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Previous optimizer run directory whose context files are copied forward.",
    )
    parser.add_argument("--root-voxel", type=float, default=1.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help=(
            "Total update cap; in coarse-to-fine mode it is split across the "
            "two stages. Each stage stops earlier on pose-update convergence "
            "or an objective-plateau heuristic (default cap: 20)."
        ),
    )
    parser.add_argument("--downsample-leaf", type=float, default=0.2)
    parser.add_argument("--plane-thickness", type=float, default=0.05)
    parser.add_argument(
        "--coarse-plane-thickness",
        type=float,
        default=0.12,
        help=(
            "Coarse-phase plane thickness; wide enough to associate 10-20 cm "
            "duplicated same-surface layers."
        ),
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Disable the default coarse-to-fine schedule and run one stage at --plane-thickness.",
    )
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--max-range", type=float, default=80.0)
    surface_mode = parser.add_mutually_exclusive_group()
    surface_mode.add_argument(
        "--double-sided",
        dest="double_sided",
        action="store_true",
        help="Preserve independently observed faces of thin structure (default).",
    )
    surface_mode.add_argument(
        "--no-double-sided",
        dest="double_sided",
        action="store_false",
        help="Ablation: use a single-surface model that may collapse thin walls.",
    )
    parser.set_defaults(double_sided=True)
    parser.add_argument(
        "--double-sided-min-separation",
        type=float,
        default=0.04,
        help="Minimum face separation for observation-side branching (metres).",
    )
    parser.add_argument(
        "--pose-prior-translation-weight", type=float, default=0.0,
        help=(
            "Trust weight around the input PGO translations in summed plane-"
            "residual units; zero disables the trust term."
        ),
    )
    parser.add_argument(
        "--pose-prior-rotation-weight", type=float, default=0.0,
        help=(
            "Trust weight around the input PGO rotations in summed plane-"
            "residual units; zero disables the trust term."
        ),
    )
    # How far from its own sensor an observation may be and still count toward a
    # ghost diagnosis. Pose-error ghosting is range-independent, but angular
    # noise misplaces distant returns linearly with range, so far observations
    # of one surface masquerade as two layers -- a smaller value is STRICTER.
    #
    # It has to come from the caller. Both entry points already diagnose their
    # pre-seed baseline at the environment preset's value (20 m indoor, 30 m
    # outdoor) while this ran at a hardcoded 20, so outdoors every round after
    # the first measured the map with a tighter filter than the baseline it was
    # compared against -- and the repair check read a constraint's "before" off
    # this report and its "after" off a 30 m survey, which makes an unchanged
    # ghost look like it grew.
    parser.add_argument("--max-observation-range", type=float, default=20.0)
    parser.add_argument(
        "--diagnose-map-inconsistency",
        action="store_true",
        help=(
            "Optional post-refinement scan for double-layer map "
            "inconsistencies. Disabled by default; it never changes poses or "
            "adds loop constraints."
        ),
    )
    return parser


def _write_tum(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with path.open("w", encoding="utf-8") as stream:
        for index in range(poses.shape[0]):
            x, y, z = poses[index, :3, 3]
            qx, qy, qz, qw = quats[index]
            stream.write(
                f"{timestamps[index]:.9f} {x:.6f} {y:.6f} {z:.6f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def _rewrite_g2o_vertices(source: Path, target: Path, poses: np.ndarray) -> int:
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    pose_count = poses.shape[0]
    replaced = 0
    lines_out = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) >= 9 and parts[0] == "VERTEX_SE3:QUAT":
                try:
                    vertex_id = int(parts[1])
                except ValueError:
                    vertex_id = -1
                if 0 <= vertex_id < pose_count:
                    x, y, z = poses[vertex_id, :3, 3]
                    qx, qy, qz, qw = quats[vertex_id]
                    lines_out.append(
                        f"VERTEX_SE3:QUAT {vertex_id} {x:.6f} {y:.6f} {z:.6f} "
                        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
                    )
                    replaced += 1
                    continue
            lines_out.append(line)
    target.write_text("".join(lines_out), encoding="utf-8")
    return replaced


def _termination_summary(
    *,
    single_stage: bool,
    stage_plan: list[tuple[float, int]],
    stage_reports: list[dict],
) -> dict:
    """Describe why a successful BALM invocation stopped.

    ``BalmResult.converged`` deliberately retains its conventional numerical
    meaning. Keep process completion, pose-update convergence, an objective
    plateau, and the configured finite update cap separate instead of
    overloading one boolean.
    """
    requested_updates = int(sum(count for _, count in stage_plan))
    completed_updates = int(sum(
        len(stage.get("iterations") or []) for stage in stage_reports
    ))
    def pose_converged(stage: dict) -> bool:
        # ``converged`` was historically also true for an RMS plateau.  Require
        # the explicit reason when available while still reading old reports.
        reason = str(stage.get("stop_reason", ""))
        return bool(stage.get("converged")) and reason in {
            "pose_converged", "converged",
        }

    def objective_plateau(stage: dict) -> bool:
        return str(stage.get("stop_reason", "")) in {
            "objective_plateau", "plateau",
        }

    all_stages_numerically_converged = bool(stage_reports) and all(
        pose_converged(stage) for stage in stage_reports
    )
    final_stage = stage_reports[-1] if stage_reports else {}
    final_stage_numerically_converged = pose_converged(final_stage)
    final_stage_objective_plateau = objective_plateau(final_stage)
    if final_stage_numerically_converged:
        reason = "final_stage_pose_converged"
    elif final_stage_objective_plateau:
        reason = "final_stage_objective_plateau"
    elif final_stage.get("stop_reason") == "no_descent_step":
        reason = "final_stage_no_descent_step"
    elif completed_updates == requested_updates:
        reason = "configured_update_cap_reached"
    else:
        reason = str(
            (stage_reports[-1] if stage_reports else {}).get(
                "stop_reason", "stopped_before_update_budget"
            )
        )

    return {
        "run_status": "completed",
        "policy": (
            "single_stage_convergence" if single_stage
            else "coarse_to_fine_convergence"
        ),
        "reason": reason,
        "requested_updates": requested_updates,
        "completed_updates": completed_updates,
        "final_stage_numerically_converged": (
            final_stage_numerically_converged
        ),
        # Preferred explicit names; the numerical aliases above remain for
        # report consumers written before the terminology correction.
        "final_stage_pose_converged": final_stage_numerically_converged,
        "final_stage_objective_plateau": final_stage_objective_plateau,
        "final_stage_stop_reason": final_stage.get("stop_reason"),
        "all_stages_numerically_converged": all_stages_numerically_converged,
        "all_stages_pose_converged": all_stages_numerically_converged,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        total_iterations = max(1, int(args.max_iterations))
        if args.single_stage:
            stage_plan = [(float(args.plane_thickness), total_iterations)]
        else:
            coarse_iterations = (total_iterations + 1) // 2
            stage_plan = [
                (float(args.coarse_plane_thickness), coarse_iterations),
                (float(args.plane_thickness), total_iterations - coarse_iterations),
            ]
            stage_plan = [(thickness, count) for thickness, count in stage_plan if count > 0]

        def stage_params(thickness: float, iterations: int) -> BalmParams:
            return BalmParams(
                root_voxel_size=float(args.root_voxel),
                max_iterations=iterations,
                downsample_leaf=float(args.downsample_leaf),
                plane_thickness=thickness,
                max_layer=int(args.max_layer),
                max_range=float(args.max_range),
                double_sided_enable=bool(args.double_sided),
                double_sided_min_separation=float(
                    args.double_sided_min_separation
                ),
                pose_prior_translation_weight=float(
                    args.pose_prior_translation_weight
                ),
                pose_prior_rotation_weight=float(
                    args.pose_prior_rotation_weight
                ),
            )

        params = stage_params(*stage_plan[0])
        schedule_text = " -> ".join(
            f"{thickness * 100:.0f}cm x{count}" for thickness, count in stage_plan
        )
        print(
            f"[BALM] Loading trajectory {args.tum} "
            f"(root_voxel={params.root_voxel_size:.2f} m, schedule: {schedule_text})"
        )
        trajectory = load_tum_trajectory(args.tum)
        pcd_paths = list_numbered_pcds(args.keyframe_dir)
        if len(pcd_paths) != trajectory.size:
            raise RuntimeError(
                f"Keyframe count ({len(pcd_paths)}) does not match trajectory pose count "
                f"({trajectory.size})."
            )
        validate_keyframe_numbering(pcd_paths, trajectory.size)

        clouds = []
        total_points = 0
        for index, pcd_path in enumerate(pcd_paths):
            cloud = prepare_local_cloud(load_xyz_points(pcd_path), params)
            clouds.append(cloud)
            total_points += cloud.shape[0]
            if (index + 1) % 100 == 0 or index + 1 == len(pcd_paths):
                print(f"[BALM] Loaded {index + 1}/{len(pcd_paths)} keyframes ({total_points} pts)")

        poses = trajectory.transforms_world_sensor
        pgo_reference_poses = poses.copy()
        stage_reports = []
        result = None
        for stage_index, (thickness, iterations) in enumerate(stage_plan, start=1):
            if len(stage_plan) > 1:
                print(
                    f"[BALM] Stage {stage_index}/{len(stage_plan)}: "
                    f"plane_thickness={thickness * 100:.0f} cm, iterations={iterations}"
                )
            result = run_balm_refinement(
                clouds,
                poses,
                stage_params(thickness, iterations),
                log_fn=print,
                reference_poses_world_sensor=pgo_reference_poses,
            )
            poses = result.poses_world_sensor
            stage_reports.append(
                {"plane_thickness": thickness, **result.report_dict()}
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_tum = args.output_dir / "optimized_poses_tum.txt"
        _write_tum(output_tum, trajectory.timestamps, result.poses_world_sensor)
        print(f"[BALM] Refined trajectory written to {output_tum}")

        ghost_regions_all = []
        ghost_regions = []
        if args.diagnose_map_inconsistency:
            print(
                "[BALM] Optional duplicated-surface audit enabled; "
                "scanning the refined map ..."
            )
            # Detected uncapped, then reported twice. `ghost_regions` stays the
            # ten worst for operator inspection; `ghost_regions_all` preserves
            # complete research-mode evidence. Neither list changes poses.
            ghost_regions_all = detect_ghost_regions(
                clouds,
                result.poses_world_sensor,
                stage_params(*stage_plan[-1]),
                log_fn=print,
                max_regions=100000,
                max_observation_range=float(args.max_observation_range),
            )
            ghost_regions = ghost_regions_all[:10]
            if not ghost_regions:
                print("[BALM] No significant map inconsistencies detected.")
        else:
            print(
                "[BALM] Duplicated-surface audit disabled (production "
                "default); poses are unaffected."
            )

        replaced_vertices = 0
        if args.source_run_dir is not None and args.source_run_dir.is_dir():
            source_g2o = args.source_run_dir / "pose_graph.g2o"
            if source_g2o.is_file():
                replaced_vertices = _rewrite_g2o_vertices(
                    source_g2o,
                    args.output_dir / "pose_graph.g2o",
                    result.poses_world_sensor,
                )
                print(f"[BALM] pose_graph.g2o carried forward ({replaced_vertices} vertices updated)")
            for name in ("manual_loop_report.json", "manual_loop_constraints.csv"):
                source_file = args.source_run_dir / name
                if source_file.is_file():
                    shutil.copy2(source_file, args.output_dir / name)

        termination = _termination_summary(
            single_stage=bool(args.single_stage),
            stage_plan=stage_plan,
            stage_reports=stage_reports,
        )
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_status": termination["run_status"],
            "numerically_converged": termination[
                "final_stage_numerically_converged"
            ],
            "pose_converged": termination["final_stage_pose_converged"],
            "objective_plateau": termination[
                "final_stage_objective_plateau"
            ],
            "input_tum": str(args.tum),
            "source_run_dir": str(args.source_run_dir) if args.source_run_dir else None,
            "pose_count": trajectory.size,
            "downsampled_points": total_points,
            "updated_g2o_vertices": replaced_vertices,
            "params": {
                "solver": "joint_plane_schur_lm",
                "root_voxel_size": params.root_voxel_size,
                "max_layer": params.max_layer,
                "min_voxel_points": params.min_voxel_points,
                "stage_schedule": [
                    {"plane_thickness": thickness, "iterations": count}
                    for thickness, count in stage_plan
                ],
                "downsample_leaf": params.downsample_leaf,
                "max_range": params.max_range,
                "max_observation_range": float(args.max_observation_range),
                "map_inconsistency_diagnostics_enabled": bool(
                    args.diagnose_map_inconsistency
                ),
                "map_inconsistency_diagnostic_scope": (
                    "same-side_duplicated_planar_surfaces"
                ),
                "diagnostic_ranking": (
                    "projected_overlap_area_times_layer_separation"
                ),
                "max_iterations": total_iterations,
                "reassociate_every": params.reassociate_every,
                "reassociate_min_motion": params.reassociate_min_motion,
                "initial_lm_damping": params.lm_damping,
                "translation_convergence_tol_m": params.convergence_tol,
                "rotation_convergence_tol_deg": float(
                    np.degrees(params.convergence_tol)
                ),
                "robust_kernel": params.robust_kernel,
                "robust_delta_scale": params.robust_delta_scale,
                "plateau_window": params.plateau_window,
                "plateau_min_improvement": params.plateau_min_improvement,
                "double_sided_enable": params.double_sided_enable,
                "double_sided_min_separation": (
                    params.double_sided_min_separation
                ),
                "pose_prior_translation_weight": (
                    params.pose_prior_translation_weight
                ),
                "pose_prior_rotation_weight": (
                    params.pose_prior_rotation_weight
                ),
            },
            "stages": stage_reports,
            "termination": termination,
            "ghost_regions": ghost_regions,
            "ghost_regions_all": ghost_regions_all,
            **result.report_dict(),
        }
        (args.output_dir / "balm_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

        all_iteration_stats = [
            item for stage in stage_reports for item in stage["iterations"]
        ]
        total_elapsed = sum(stage["elapsed_sec"] for stage in stage_reports)
        if all_iteration_stats:
            print(
                f"[BALM] Finished: rms {all_iteration_stats[0]['rms_plane_distance']:.4f} -> "
                f"{all_iteration_stats[-1]['rms_plane_distance']:.4f} m over "
                f"{len(all_iteration_stats)} iterations in {len(stage_reports)} stage(s) "
                f"({termination['reason']}), "
                f"elapsed={total_elapsed:.1f}s"
            )
        else:
            print("[BALM] Finished without iterations (no planar structure found); poses unchanged.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[BALM] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
# ===== END CHANGE: balm refinement cli =====
