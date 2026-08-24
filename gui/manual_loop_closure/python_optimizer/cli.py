#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
GUI_DIR = CURRENT_DIR.parents[1]
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure.optimizer_backend import (  # noqa: E402
    OPTIMIZE_MODE_ISAM2,
    OPTIMIZE_MODE_GNC_TLS,
    OPTIMIZE_MODE_LM_CHORDAL,
    OPTIMIZE_MODE_LM_HUBER,
    OPTIMIZE_MODE_LM,
    OptimizerRunOptions,
)
from manual_loop_closure.python_optimizer.optimizer import run_python_optimizer  # noqa: E402


# ===== BEGIN CHANGE: python optimizer cli =====
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure Python optimizer backend for LiDAR Map Refiner."
    )
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--g2o", required=True, type=Path)
    parser.add_argument("--tum", required=True, type=Path)
    parser.add_argument("--keyframe-dir", required=True, type=Path)
    parser.add_argument("--constraints-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--map-voxel-leaf", required=False, type=float, default=0.2)
    parser.add_argument(
        "--optimize-mode",
        required=False,
        choices=(OPTIMIZE_MODE_LM, OPTIMIZE_MODE_ISAM2,
                 OPTIMIZE_MODE_LM_HUBER, OPTIMIZE_MODE_GNC_TLS,
                 OPTIMIZE_MODE_LM_CHORDAL),
        default=OPTIMIZE_MODE_LM,
    )
    parser.add_argument("--skip-map-build", action="store_true")
    parser.add_argument("--skip-graph-plot", action="store_true")
    parser.add_argument("--manual-information-scale", type=float, default=1.0)
    parser.add_argument("--loop-correlation-window-keyframes", type=int, default=0)
    parser.add_argument("--loop-cluster-information-budget", type=float, default=0.0)
    parser.add_argument(
        "--loop-correlation-policy",
        choices=("pair", "shared_endpoint"), default="pair",
    )
    parser.add_argument(
        "--loop-cluster-allocation",
        choices=("equal", "representative"), default="equal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    options = OptimizerRunOptions(
        session_root=args.session_root,
        g2o_path=args.g2o,
        tum_path=args.tum,
        keyframe_dir=args.keyframe_dir,
        constraints_csv=args.constraints_csv,
        output_dir=args.output_dir,
        map_voxel_leaf=float(args.map_voxel_leaf),
        optimize_mode=str(args.optimize_mode),
        skip_map_build=bool(args.skip_map_build),
        manual_information_scale=float(args.manual_information_scale),
        loop_correlation_window_keyframes=int(
            args.loop_correlation_window_keyframes
        ),
        loop_cluster_information_budget=float(
            args.loop_cluster_information_budget
        ),
        loop_correlation_policy=str(args.loop_correlation_policy),
        loop_cluster_allocation=str(args.loop_cluster_allocation),
        skip_graph_plot=bool(args.skip_graph_plot),
    )
    try:
        result = run_python_optimizer(options, log_fn=print)
    except Exception as exc:
        print(f"[PythonOptimizer] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  output_dir: {result.output_dir}")
    print(f"  enabled_constraints: {result.enabled_constraints}")
    print(f"  pose_count: {result.pose_count}")
    print(f"  factor_count: {result.factor_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ===== END CHANGE: python optimizer cli =====
