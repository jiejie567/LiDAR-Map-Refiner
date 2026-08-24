from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from manual_loop_closure.optimizer_backend import (
    OPTIMIZE_MODE_ISAM2,
    OPTIMIZE_MODE_GNC_TLS,
    OPTIMIZE_MODE_LM_CHORDAL,
    OPTIMIZE_MODE_LM_HUBER,
    OPTIMIZE_MODE_LM,
    OptimizerRunOptions,
    OptimizerRunResult,
)
from manual_loop_closure.trajectory_io import load_tum_trajectory

from .exporters import (
    MeasurementRecord,
    build_optimized_map,
    generate_pose_graph_png,
    save_report_json,
    save_trajectory_pcd,
    save_tum,
    save_xyzi_pcd,
    write_pose_graph_g2o,
)
from .graph_loader import BetweenFactorRecord, build_factor_graph, pose_key
from .information_order import g2o_information_to_gtsam, gtsam_information_to_g2o
from .loop_weighting import (
    CORRELATION_CLUSTERING_METHOD,
    LoopFactorWeightInput,
    weight_loop_information,
)


LogFn = Optional[Callable[[str], None]]


# ===== BEGIN CHANGE: python optimizer orchestration =====
@dataclass(frozen=True)
class ManualConstraintSpec:
    enabled: bool
    source_id: int
    target_id: int
    translation_xyz: np.ndarray
    quat_xyzw: np.ndarray
    sigma_t_xyz: np.ndarray
    sigma_r_deg: np.ndarray
    information_g2o: np.ndarray | None = None
    sigma_rotation_unit: str = "deg"
    confidence: float = 1.0
    factor_information_scale: float = 1.0


def _log(log_fn: LogFn, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _import_gtsam():
    try:
        import gtsam  # type: ignore
    except Exception as exc:  # pragma: no cover - import behavior is environment-specific
        raise RuntimeError(
            "Python GTSAM is unavailable. Install the Python wrapper first "
            "(see docs/INSTALL_GTSAM_PYTHON.md)."
        ) from exc
    return gtsam


def _load_measurements(tum_path: Path, keyframe_dir: Path) -> list[MeasurementRecord]:
    trajectory = load_tum_trajectory(tum_path)
    cloud_paths = sorted(
        [
            entry
            for entry in keyframe_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".pcd" and entry.stem.isdigit()
        ],
        key=lambda item: int(item.stem),
    )
    if len(cloud_paths) != trajectory.size:
        raise RuntimeError(
            "Keyframe count does not match optimized_poses_tum.txt: "
            f"pcd={len(cloud_paths)} tum={trajectory.size}"
        )
    for expected_index, path in enumerate(cloud_paths):
        actual_index = int(path.stem)
        if actual_index != expected_index:
            raise RuntimeError(
                "Keyframe numbering must match 0..N-1 exactly. "
                f"Expected {expected_index}.pcd but found {path.name}"
            )
    measurements = [
        MeasurementRecord(index=index, odom_time=float(trajectory.timestamps[index]), cloud_path=cloud_paths[index])
        for index in range(trajectory.size)
    ]
    return measurements


def _parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"Invalid bool value in constraints csv: {text}")


def _load_constraints_csv(path: Path) -> list[ManualConstraintSpec]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = [
            "enabled",
            "source_id",
            "target_id",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
            "sigma_tx",
            "sigma_ty",
            "sigma_tz",
            "sigma_roll_deg",
            "sigma_pitch_deg",
            "sigma_yaw_deg",
        ]
        if reader.fieldnames is None:
            raise RuntimeError(f"Constraints CSV is empty: {path}")
        for field in required:
            if field not in reader.fieldnames:
                raise RuntimeError(f"Constraints CSV missing required column: {field}")

        constraints: list[ManualConstraintSpec] = []
        for row in reader:
            if row is None:
                continue
            information_g2o = None
            information_text = row.get("information_upper_json")
            if information_text:
                values = json.loads(information_text)
                if not isinstance(values, list) or len(values) != 21:
                    raise RuntimeError(
                        "information_upper_json must contain 21 upper-triangle values"
                    )
                information_g2o = np.zeros((6, 6), dtype=np.float64)
                cursor = 0
                for matrix_row in range(6):
                    for matrix_col in range(matrix_row, 6):
                        value = float(values[cursor])
                        information_g2o[matrix_row, matrix_col] = value
                        information_g2o[matrix_col, matrix_row] = value
                        cursor += 1
                eigenvalues = np.linalg.eigvalsh(information_g2o)
                if not np.isfinite(information_g2o).all() or eigenvalues[0] <= 0.0:
                    raise RuntimeError(
                        "information_upper_json must encode a finite SPD matrix"
                    )
            constraints.append(
                ManualConstraintSpec(
                    enabled=_parse_bool(row["enabled"]),
                    source_id=int(row["source_id"]),
                    target_id=int(row["target_id"]),
                    translation_xyz=np.asarray(
                        [row["tx"], row["ty"], row["tz"]],
                        dtype=np.float64,
                    ),
                    quat_xyzw=np.asarray(
                        [row["qx"], row["qy"], row["qz"], row["qw"]],
                        dtype=np.float64,
                    ),
                    sigma_t_xyz=np.asarray(
                        [row["sigma_tx"], row["sigma_ty"], row["sigma_tz"]],
                        dtype=np.float64,
                    ),
                    sigma_r_deg=np.asarray(
                        [row["sigma_roll_deg"], row["sigma_pitch_deg"], row["sigma_yaw_deg"]],
                        dtype=np.float64,
                    ),
                    information_g2o=information_g2o,
                    sigma_rotation_unit=(
                        row.get("sigma_rotation_unit") or "deg"
                    ).strip().lower(),
                    confidence=float(row.get("confidence") or 1.0),
                    factor_information_scale=float(
                        row.get("factor_information_scale") or 1.0
                    ),
                )
            )
    return constraints


def _manual_constraint_information(constraint: ManualConstraintSpec) -> np.ndarray:
    if constraint.information_g2o is not None:
        return g2o_information_to_gtsam(constraint.information_g2o)
    if constraint.sigma_rotation_unit == "deg":
        sigma_r_rad = np.deg2rad(constraint.sigma_r_deg)
    elif constraint.sigma_rotation_unit == "rad":
        sigma_r_rad = np.asarray(constraint.sigma_r_deg, dtype=np.float64)
    else:
        raise RuntimeError(
            "sigma_rotation_unit must be either 'deg' or 'rad', got "
            f"{constraint.sigma_rotation_unit!r}"
        )
    sigmas = np.concatenate([sigma_r_rad, constraint.sigma_t_xyz], axis=0)
    information = np.zeros((6, 6), dtype=np.float64)
    for idx, sigma in enumerate(sigmas):
        if sigma <= 0.0:
            raise RuntimeError("Constraint sigmas must be positive.")
        information[idx, idx] = 1.0 / (sigma * sigma)
    return information


def _normalize_optimize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == OPTIMIZE_MODE_ISAM2:
        return OPTIMIZE_MODE_ISAM2
    if normalized == OPTIMIZE_MODE_GNC_TLS:
        return OPTIMIZE_MODE_GNC_TLS
    if normalized == OPTIMIZE_MODE_LM_HUBER:
        return OPTIMIZE_MODE_LM_HUBER
    if normalized == OPTIMIZE_MODE_LM_CHORDAL:
        return OPTIMIZE_MODE_LM_CHORDAL
    return OPTIMIZE_MODE_LM


def _solve_graph(
    *,
    gtsam,
    graph,
    initial_values,
    optimize_mode: str,
    known_inlier_count: int,
    log_fn: LogFn,
):
    mode = _normalize_optimize_mode(optimize_mode)
    original_initial_error = float(graph.error(initial_values))
    if mode == OPTIMIZE_MODE_LM_CHORDAL:
        # Chordal orientation initialization separates optimizer basin failures
        # from measurement/covariance failures.  It is deliberately opt-in:
        # well-initialized odometry graphs normally do not need it.
        initial_values = gtsam.InitializePose3.initialize(
            graph, initial_values, False
        )
    initial_error = float(graph.error(initial_values))
    _log(
        log_fn,
        f"[PythonOptimizer] Starting {mode.upper()} optimize factors={int(graph.size())}, initial_error={initial_error:.9e}",
    )
    if mode == OPTIMIZE_MODE_LM_CHORDAL:
        _log(
            log_fn,
            "[PythonOptimizer] Chordal initialization "
            f"original_error={original_initial_error:.9e}, "
            f"initialized_error={initial_error:.9e}",
        )
    stage_start = time.perf_counter()
    robust_weights = None
    if mode == OPTIMIZE_MODE_ISAM2:
        params = gtsam.ISAM2Params()
        if hasattr(params, "setFactorization"):
            params.setFactorization("CHOLESKY")
        if hasattr(params, "setRelinearizeThreshold"):
            params.setRelinearizeThreshold(0.01)
        if hasattr(params, "relinearizeSkip"):
            params.relinearizeSkip = 1
        isam = gtsam.ISAM2(params)
        isam.update(graph, initial_values)
        for _ in range(4):
            isam.update()
        optimized = isam.calculateEstimate()
        solve_label = "ISAM2"
    elif mode == OPTIMIZE_MODE_GNC_TLS:
        base = gtsam.LevenbergMarquardtParams()
        base.setMaxIterations(100)
        params = gtsam.GncLMParams(base)
        params.setLossType(gtsam.GncLossType.TLS)
        params.setKnownInliers(list(range(int(known_inlier_count))))
        optimizer = gtsam.GncLMOptimizer(graph, initial_values, params)
        optimized = optimizer.optimize()
        robust_weights = np.asarray(optimizer.getWeights(), dtype=float).reshape(-1)
        solve_label = "GNC-TLS"
    else:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(50)
        if hasattr(params, "setVerbosityLM"):
            params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            graph,
            initial_values,
            params,
        )
        optimized = optimizer.optimize()
        solve_label = "LM-CHORDAL" if mode == OPTIMIZE_MODE_LM_CHORDAL else "LM"
    optimize_elapsed = time.perf_counter() - stage_start
    final_error = float(graph.error(optimized))
    _log(
        log_fn,
        f"[PythonOptimizer] {solve_label} finished "
        f"elapsed={optimize_elapsed:.2f}s, initial_error={initial_error:.9e}, "
        f"final_error={final_error:.9e}, improvement={initial_error - final_error:.9e}",
    )
    return optimized, initial_error, final_error, optimize_elapsed, mode, robust_weights


def run_python_optimizer(
    options: OptimizerRunOptions,
    log_fn: LogFn = None,
) -> OptimizerRunResult:
    total_start = time.perf_counter()
    gtsam = _import_gtsam()
    _log(log_fn, f"[PythonOptimizer] Using gtsam from {Path(gtsam.__file__).resolve()}")

    stage_start = time.perf_counter()
    measurements = _load_measurements(options.tum_path, options.keyframe_dir)
    _log(
        log_fn,
        f"[PythonOptimizer] Loaded measurements poses={len(measurements)}, elapsed={time.perf_counter() - stage_start:.2f}s",
    )

    stage_start = time.perf_counter()
    graph_result = build_factor_graph(
        session_root=options.session_root,
        g2o_path=options.g2o_path,
        gtsam_mod=gtsam,
        log_fn=log_fn,
    )
    _log(
        log_fn,
        "[PythonOptimizer] Factor graph ready "
        f"poses={graph_result.pose_count}, factors={int(graph_result.graph.size())}, "
        f"elapsed={time.perf_counter() - stage_start:.2f}s",
    )
    if graph_result.pose_count != len(measurements):
        raise RuntimeError(
            "Pose graph and measurements disagree on pose count: "
            f"graph={graph_result.pose_count} measurements={len(measurements)}"
        )

    stage_start = time.perf_counter()
    base_factor_count = int(graph_result.graph.size())
    constraints = _load_constraints_csv(options.constraints_csv)
    _log(
        log_fn,
        f"[PythonOptimizer] Loaded manual constraints total={len(constraints)}, elapsed={time.perf_counter() - stage_start:.2f}s",
    )
    enabled_constraints = 0
    factor_records = list(graph_result.factor_records)
    active_constraints = [constraint for constraint in constraints if constraint.enabled]
    weighting = weight_loop_information(
        [
            LoopFactorWeightInput(
                source_id=constraint.source_id,
                target_id=constraint.target_id,
                information=_manual_constraint_information(constraint),
                confidence=constraint.confidence,
                factor_scale=constraint.factor_information_scale,
            )
            for constraint in active_constraints
        ],
        information_scale=options.manual_information_scale,
        correlation_window_keyframes=options.loop_correlation_window_keyframes,
        cluster_information_budget=options.loop_cluster_information_budget,
        correlation_policy=options.loop_correlation_policy,
        cluster_allocation=options.loop_cluster_allocation,
    )
    weighting_records = []
    active_index = 0
    for constraint in constraints:
        if not constraint.enabled:
            continue
        if (
            constraint.source_id < 0
            or constraint.target_id < 0
            or constraint.source_id >= len(measurements)
            or constraint.target_id >= len(measurements)
        ):
            raise RuntimeError(
                "Constraint index out of range: "
                f"target={constraint.target_id} source={constraint.source_id}"
            )
        measured_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                float(constraint.quat_xyzw[3]),
                float(constraint.quat_xyzw[0]),
                float(constraint.quat_xyzw[1]),
                float(constraint.quat_xyzw[2]),
            ),
            gtsam.Point3(
                float(constraint.translation_xyz[0]),
                float(constraint.translation_xyz[1]),
                float(constraint.translation_xyz[2]),
            ),
        )
        weight = weighting[active_index]
        active_index += 1
        information_gtsam = weight.effective_information
        noise = gtsam.noiseModel.Gaussian.Information(information_gtsam)
        if _normalize_optimize_mode(options.optimize_mode) == OPTIMIZE_MODE_LM_HUBER:
            kernel = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
            noise = gtsam.noiseModel.Robust.Create(kernel, noise)
        graph_result.graph.add(
            gtsam.BetweenFactorPose3(
                pose_key(gtsam, constraint.target_id),
                pose_key(gtsam, constraint.source_id),
                measured_pose,
                noise,
            )
        )
        factor_records.append(
            BetweenFactorRecord(
                node_i=constraint.target_id,
                node_j=constraint.source_id,
                translation_xyz=constraint.translation_xyz,
                quat_xyzw=constraint.quat_xyzw,
                information=gtsam_information_to_g2o(information_gtsam),
                origin="manual",
            )
        )
        weighting_records.append({
            "source_id": int(constraint.source_id),
            "target_id": int(constraint.target_id),
            "sigma_rotation_unit": constraint.sigma_rotation_unit,
            "confidence": float(constraint.confidence),
            "factor_information_scale": float(
                constraint.factor_information_scale
            ),
            "cluster_id": int(weight.cluster_id),
            "cluster_size": int(weight.cluster_size),
            "correlation_scale": float(weight.correlation_scale),
            "global_scale": float(weight.global_scale),
            "applied_scale": float(weight.applied_scale),
            "information_eigenvalues": np.linalg.eigvalsh(
                information_gtsam
            ).tolist(),
        })
        enabled_constraints += 1

    _log(
        log_fn,
        f"[PythonOptimizer] Active manual constraints enabled={enabled_constraints}",
    )
    if enabled_constraints == 0:
        _log(
            log_fn,
            "[PythonOptimizer] No enabled manual constraints found in CSV. "
            "Proceeding with the filtered input graph only.",
        )

    (optimized, initial_error, final_error, optimize_elapsed, optimize_mode,
     robust_weights) = _solve_graph(
        gtsam=gtsam,
        graph=graph_result.graph,
        initial_values=graph_result.initial_values,
        optimize_mode=options.optimize_mode,
        known_inlier_count=base_factor_count,
        log_fn=log_fn,
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    output_g2o = options.output_dir / "pose_graph.g2o"
    output_tum = options.output_dir / "optimized_poses_tum.txt"
    output_map = options.output_dir / "global_map_manual_imu.pcd"
    output_trajectory = options.output_dir / "trajectory.pcd"
    output_png = options.output_dir / "pose_graph.png"
    output_report = options.output_dir / "manual_loop_report.json"
    copied_constraints = options.output_dir / "manual_loop_constraints.csv"
    map_built = False
    map_build_elapsed_sec = 0.0
    map_point_count = 0

    write_pose_graph_g2o(
        output_g2o,
        optimized,
        graph_result.pose_count,
        factor_records,
        gtsam,
    )
    save_tum(output_tum, measurements, optimized, gtsam)
    if options.skip_map_build:
        _log(log_fn, "[PythonOptimizer] Map rebuild deferred until export.")
        optimized_map = None
    else:
        map_stage_start = time.perf_counter()
        optimized_map = build_optimized_map(
            measurements,
            optimized,
            gtsam,
            options.map_voxel_leaf,
            log_fn=log_fn,
        )
        if optimized_map.size == 0:
            raise RuntimeError("Optimized point cloud map is empty.")
        save_xyzi_pcd(output_map, optimized_map)
        save_trajectory_pcd(output_trajectory, measurements, optimized, gtsam)
        map_built = True
        map_build_elapsed_sec = time.perf_counter() - map_stage_start
        map_point_count = int(optimized_map.shape[0])
    if not options.skip_graph_plot:
        generate_pose_graph_png(
            Path(__file__).resolve().parents[3], output_g2o, output_png,
            log_fn=log_fn,
        )
    save_report_json(
        output_report,
        session_root=options.session_root,
        input_g2o=options.g2o_path,
        input_tum=options.tum_path,
        input_keyframe_dir=options.keyframe_dir,
        constraints_csv=options.constraints_csv,
        output_dir=options.output_dir,
        map_voxel_leaf=options.map_voxel_leaf,
        optimize_mode=optimize_mode,
        total_constraints=len(constraints),
        enabled_constraints=enabled_constraints,
        optimized_pose_count=len(measurements),
        factor_count=int(graph_result.graph.size()),
        map_point_count=map_point_count,
        map_built=map_built,
        map_build_elapsed_sec=map_build_elapsed_sec,
        robust_weights=(
            None if robust_weights is None else robust_weights.tolist()
        ),
        manual_factor_weighting={
            "information_scale": float(options.manual_information_scale),
            "correlation_window_keyframes": int(
                options.loop_correlation_window_keyframes
            ),
            "cluster_information_budget": float(
                options.loop_cluster_information_budget
            ),
            "correlation_policy": options.loop_correlation_policy,
            "correlation_clustering": CORRELATION_CLUSTERING_METHOD,
            "cluster_allocation": options.loop_cluster_allocation,
            "factors": weighting_records,
        },
    )
    if options.constraints_csv.resolve() != copied_constraints.resolve():
        shutil.copyfile(options.constraints_csv, copied_constraints)

    total_elapsed = time.perf_counter() - total_start
    _log(
        log_fn,
        "[PythonOptimizer] Finished successfully "
        f"total_elapsed={total_elapsed:.2f}s, final_error={final_error:.9e}, "
        f"map_points={map_point_count}",
    )
    return OptimizerRunResult(
        output_dir=options.output_dir,
        output_g2o=output_g2o,
        output_tum=output_tum,
        output_map_pcd=output_map,
        output_trajectory_pcd=output_trajectory,
        output_report_json=output_report,
        factor_count=int(graph_result.graph.size()),
        pose_count=len(measurements),
        enabled_constraints=enabled_constraints,
        map_built=map_built,
    )
# ===== END CHANGE: python optimizer orchestration =====
