#!/usr/bin/env python3
"""Run frozen pose-graph factor ablations from identical inputs and loop set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
GUI = HERE.parents[2] / "gui"
for path in (HERE.parent, GUI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manual_loop_closure.optimizer_backend import (  # noqa: E402
    OPTIMIZE_MODE_LM,
    OptimizerRunOptions,
)
from manual_loop_closure.python_optimizer.optimizer import (  # noqa: E402
    run_python_optimizer,
)
from manual_loop_closure.python_optimizer.oriented_surface_factors import (  # noqa: E402
    OrientedFactorConfig,
)
from rebuild_oriented_pose_graph import (  # noqa: E402
    VARIANTS,
    rebuild,
    sha256_file,
)


EVALUATOR = REPO / "experiments" / "mcd_new_sequences" / "evaluate_trajectory.py"
POSITION_EVALUATOR = (
    REPO / "tools" / "manual_loop_closure" / "scripts" / "ghostloop_eval" / "ate_eval.py"
)
POSITION_SEQUENCE = {"hall02": "hall_02", "hall04": "hall_04"}


def trajectory_metrics(estimate: Path, session_manifest: dict) -> dict | None:
    truth = Path(session_manifest["truth"])
    if session_manifest.get("truth_kind") == "position":
        sequence = POSITION_SEQUENCE.get(session_manifest["dataset"])
        if sequence is None:
            return None
        completed = subprocess.run(
            [sys.executable, str(POSITION_EVALUATOR), str(estimate),
             str(truth), sequence],
            capture_output=True, text=True,
        )
        try:
            rmse_m = float(completed.stdout.split("=")[1].split("cm")[0]) / 100.0
        except (ValueError, IndexError):
            return None
        return {"position_error_m": {"rmse": rmse_m}, "rotation_error_deg": None}
    completed = subprocess.run(
        [sys.executable, str(EVALUATOR), "--estimate", str(estimate),
         "--ground-truth", str(truth)],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def run(
    session: Path,
    loop_csv: Path,
    empty_constraints_csv: Path,
    output_dir: Path,
    variants: list[str],
    config: OrientedFactorConfig,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    session_manifest = json.loads((session / "input_manifest.json").read_text())
    if session_manifest.get("truth_kind") not in {"se3", "position"}:
        raise ValueError("ablation runner requires SE(3) or position truth")
    rows = []
    for variant in variants:
        variant_dir = output_dir / variant
        graph_dir = variant_dir / "rebuilt_graph"
        variant_config = config
        if variant in {
            "icp_isotropic", "icp_anisotropic", "icp_anisotropic_reweight",
            "loop_icp_isotropic", "loop_icp_anisotropic",
        }:
            variant_config = OrientedFactorConfig(
                **{**asdict(config), "normal_gate_deg": 180.0}
            )
        rebuild_manifest = rebuild(
            session / "pose_graph.g2o",
            session / "key_point_frame",
            graph_dir,
            variant,
            variant_config,
            loop_csv,
        )
        optimizer_dir = variant_dir / "optimized"
        result = run_python_optimizer(
            OptimizerRunOptions(
                session_root=session,
                g2o_path=graph_dir / "pose_graph.g2o",
                tum_path=session / "optimized_poses_tum.txt",
                keyframe_dir=session / "key_point_frame",
                constraints_csv=empty_constraints_csv,
                output_dir=optimizer_dir,
                map_voxel_leaf=0.2,
                optimize_mode=OPTIMIZE_MODE_LM,
                skip_map_build=True,
            ),
            log_fn=print,
        )
        metrics = trajectory_metrics(result.output_tum, session_manifest)
        rows.append({
            "variant": variant,
            "sequential_replaced": rebuild_manifest["counts"]["sequential_replaced"],
            "sequential_fallback": rebuild_manifest["counts"]["sequential_fallback"],
            "sequential_reweighted": rebuild_manifest["counts"].get(
                "sequential_reweighted", 0
            ),
            "loop_proposed": rebuild_manifest["counts"]["loop_proposed"],
            "loop_added": rebuild_manifest["counts"]["loop_added"],
            "position_ate_rmse_m": (
                metrics["position_error_m"]["rmse"] if metrics else None
            ),
            "rotation_ate_rmse_deg": (
                metrics["rotation_error_deg"]["rmse"]
                if metrics and metrics["rotation_error_deg"] else None
            ),
            "graph_sha256": sha256_file(graph_dir / "pose_graph.g2o"),
            "trajectory_sha256": sha256_file(result.output_tum),
        })
    csv_path = output_dir / "ablation_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": session_manifest["dataset"],
        "controlled_inputs": {
            "session": str(session),
            "pose_graph_sha256": sha256_file(session / "pose_graph.g2o"),
            "trajectory_sha256": sha256_file(session / "optimized_poses_tum.txt"),
            "loop_csv_sha256": sha256_file(loop_csv),
            "empty_constraints_sha256": sha256_file(empty_constraints_csv),
        },
        "factor_config": asdict(config),
        "variants": rows,
        "source_data_sha256": sha256_file(csv_path),
    }
    (output_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--loop-csv", type=Path, required=True)
    parser.add_argument("--empty-constraints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--information-scale", type=float, required=True)
    parser.add_argument("--voxel", type=float, default=0.20)
    parser.add_argument("--normal-radius", type=float, default=0.60)
    parser.add_argument("--max-correspondence", type=float, default=0.60)
    parser.add_argument("--normal-gate-deg", type=float, default=45.0)
    parser.add_argument("--min-overlap", type=float, default=0.20)
    parser.add_argument("--min-correspondences", type=int, default=80)
    parser.add_argument("--translation-sigma-floor-m", type=float, default=0.0)
    parser.add_argument("--rotation-sigma-floor-deg", type=float, default=0.0)
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    config = OrientedFactorConfig(
        voxel_size_m=args.voxel,
        normal_radius_m=args.normal_radius,
        max_correspondence_m=args.max_correspondence,
        normal_gate_deg=args.normal_gate_deg,
        min_overlap=args.min_overlap,
        min_correspondences=args.min_correspondences,
        information_scale=args.information_scale,
        translation_sigma_floor_m=args.translation_sigma_floor_m,
        rotation_sigma_floor_rad=math.radians(args.rotation_sigma_floor_deg),
    )
    report = run(
        args.session.expanduser().resolve(),
        args.loop_csv.expanduser().resolve(),
        args.empty_constraints.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        variants,
        config,
    )
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
