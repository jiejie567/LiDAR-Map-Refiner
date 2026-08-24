"""Joint plane bundle adjustment for keyframe pose refinement.

Adaptive voxelization extracts planar factors from the aggregated map. Each
plane's two normal coordinates and offset are analytically eliminated from a
sparse joint pose system, and LM accepts only objective-decreasing steps. Pure
numpy/scipy, no GTSAM or ROS required.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, cg, spsolve
from scipy.spatial.transform import Rotation


LogFn = Optional[Callable[[str], None]]


# Sparse direct factorization can require orders of magnitude more memory than
# the Schur matrix itself because of fill-in.  The 2579-pose IH graph has 15468
# variables and exhausted a 30 GiB workstation inside UMFPACK before Python
# could raise an exception.  Above this size use preconditioned CG on the same
# damped normal equations: this changes only the linear algebra backend, not the
# BALM factors, objective, gauge anchor, or accepted-step test.
_DIRECT_SOLVER_MAX_VARIABLES = 12_000


def _log(log_fn: LogFn, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


# ===== BEGIN CHANGE: balm plane bundle adjustment =====
@dataclass(frozen=True)
class BalmParams:
    root_voxel_size: float = 1.0
    max_layer: int = 3
    min_voxel_points: int = 20
    min_poses_per_voxel: int = 2
    plane_thickness: float = 0.05
    plane_spread_ratio: float = 2.0
    downsample_leaf: float = 0.4
    max_range: float = 80.0
    max_points_per_voxel_pose: int = 40
    # Total coarse/fine update cap. Production stops on pose-update convergence
    # or an RMS plateau; this number is only a finite-runtime safeguard. The
    # historical two-update operating point remains reproducible by explicitly
    # setting max_iterations=2, but is not a defensible deployment default
    # because it was selected with ground truth unavailable at runtime.
    max_iterations: int = 20
    reassociate_every: int = 1
    # Skip re-extraction while poses have moved less than this since the last
    # association (the voxel membership cannot have changed meaningfully).
    reassociate_min_motion: float = 0.0
    # Initial diagonal LM damping.  It is adapted internally: a step is accepted
    # only when the fixed-association plane objective decreases, otherwise the
    # step is rolled back and the damping is increased.
    lm_damping: float = 1e-2
    convergence_tol: float = 1e-4
    max_rot_step_rad: float = 0.2
    max_trans_step_m: float = 1.0
    # Optional ablation only. The default solver is pure plane BA; a fixed
    # absolute prior can hide a defective geometric objective and is not a
    # substitute for the original relative odometry/loop factors.
    # Values are objective weights in the same units as the summed plane
    # residuals; zero preserves the historical unregularized implementation.
    pose_prior_translation_weight: float = 0.0
    pose_prior_rotation_weight: float = 0.0
    # Preserve physical surface identity through BA. Points on opposite faces
    # of thin structure are separated by the sign of their sensor-to-surface
    # observation relative to the local plane normal. Same-face repeated
    # observations retain the same sign and may still be refined together.
    double_sided_enable: bool = True
    double_sided_min_separation: float = 0.04
    # Robust kernel: down-weights clutter/dynamic points whose plane residuals
    # exceed robust_delta_scale * plane_thickness.
    robust_kernel: str = "huber"  # "none" | "huber" | "cauchy"
    robust_delta_scale: float = 1.5
    # Objective-based convergence safeguard.  It complements the pose-update
    # tolerance when re-association changes the local quadratic model.
    plateau_window: int = 2
    plateau_min_improvement: float = 5e-5


@dataclass
class BalmIterationStats:
    iteration: int
    plane_count: int
    point_count: int
    rms_plane_distance: float
    max_pose_update: float
    reassociated: bool
    # Keep translation and rotation in their physical units.  The legacy
    # ``max_pose_update`` is retained for report compatibility, but taking the
    # maximum of metres and radians is not a valid convergence certificate.
    max_translation_update_m: float = 0.0
    p95_translation_update_m: float = 0.0
    max_rotation_update_deg: float = 0.0
    p95_rotation_update_deg: float = 0.0
    objective: float = 0.0
    accepted: bool = True
    lm_damping: float = 0.0
    double_sided_voxel_count: int = 0
    linear_solver: str = ""
    linear_solver_iterations: int = 0
    linear_solver_relative_residual: float = 0.0


@dataclass
class BalmResult:
    poses_world_sensor: np.ndarray
    iterations: List[BalmIterationStats] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = "iteration_limit"
    elapsed_sec: float = 0.0

    def report_dict(self) -> dict:
        return {
            "solver": "joint_plane_schur_lm",
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "iterations": [
                {
                    "iteration": item.iteration,
                    "plane_count": item.plane_count,
                    "point_count": item.point_count,
                    "rms_plane_distance": round(item.rms_plane_distance, 6),
                    "max_pose_update": round(item.max_pose_update, 6),
                    "max_translation_update_m": round(
                        item.max_translation_update_m, 6
                    ),
                    "p95_translation_update_m": round(
                        item.p95_translation_update_m, 6
                    ),
                    "max_rotation_update_deg": round(
                        item.max_rotation_update_deg, 6
                    ),
                    "p95_rotation_update_deg": round(
                        item.p95_rotation_update_deg, 6
                    ),
                    "reassociated": item.reassociated,
                    "objective": round(item.objective, 9),
                    "accepted": item.accepted,
                    "lm_damping": round(item.lm_damping, 9),
                    "double_sided_voxel_count": item.double_sided_voxel_count,
                    "linear_solver": item.linear_solver,
                    "linear_solver_iterations": item.linear_solver_iterations,
                    "linear_solver_relative_residual": round(
                        item.linear_solver_relative_residual, 9
                    ),
                }
                for item in self.iterations
            ],
        }


def _rms_has_plateaued(
    rms_history: List[float], window: int, min_improvement: float,
) -> bool:
    """Return true only for a small *non-negative* RMS improvement.

    Plane associations may change between iterations, so their RMS values are
    not guaranteed to be monotone even though every LM step decreases its own
    fixed-association objective.  The previous one-sided comparison also
    classified an RMS increase as ``increase < positive_threshold`` and
    reported numerical convergence while pose updates were still centimetres.
    An increase is evidence against a plateau, never evidence for one.
    """
    if window <= 0 or len(rms_history) <= window:
        return False
    improvement = float(rms_history[-window - 1] - rms_history[-1])
    return 0.0 <= improvement < float(min_improvement)


def _solve_damped_system(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    direct_max_variables: int = _DIRECT_SOLVER_MAX_VARIABLES,
) -> tuple[np.ndarray, str, int, float]:
    """Solve one damped Schur system without unsafe large-graph fill-in.

    Small systems retain the historical sparse-direct path.  Large systems use
    Jacobi-preconditioned conjugate gradients.  LM makes the symmetric Schur
    system positive definite; a non-converged iterative solve is returned as
    NaN so the existing LM loop raises damping and retries without moving any
    pose.
    """
    matrix = matrix.tocsr()
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    variable_count = int(matrix.shape[0])
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-15)

    if variable_count <= direct_max_variables:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solution = np.asarray(spsolve(matrix.tocsc(), rhs)).reshape(-1)
            except (MatrixRankWarning, MemoryError, RuntimeError, ValueError):
                solution = np.full_like(rhs, np.nan)
        relative_residual = (
            float(np.linalg.norm(matrix @ solution - rhs)) / rhs_norm
            if np.all(np.isfinite(solution))
            else float("inf")
        )
        return solution, "sparse_direct", 0, relative_residual

    diagonal = np.maximum(np.abs(matrix.diagonal()), 1e-12)
    inverse_diagonal = 1.0 / diagonal
    preconditioner = LinearOperator(
        matrix.shape,
        matvec=lambda vector: inverse_diagonal * vector,
        dtype=np.float64,
    )
    iterations = 0

    def count_iteration(_solution: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    kwargs = dict(
        A=matrix,
        b=rhs,
        M=preconditioner,
        maxiter=min(max(1_000, variable_count // 2), 5_000),
        callback=count_iteration,
        atol=0.0,
    )
    try:
        # SciPy >=1.12 renamed ``tol`` to ``rtol``.  Supporting both keeps the
        # CLI usable in the system test environment and in the GUI virtualenv.
        try:
            solution, info = cg(**kwargs, rtol=1e-6)
        except TypeError:
            solution, info = cg(**kwargs, tol=1e-6)
    except (MemoryError, RuntimeError, ValueError):
        solution = np.full_like(rhs, np.nan)
        info = -1
    solution = np.asarray(solution).reshape(-1)
    relative_residual = (
        float(np.linalg.norm(matrix @ solution - rhs)) / rhs_norm
        if np.all(np.isfinite(solution))
        else float("inf")
    )
    if info != 0 or not np.all(np.isfinite(solution)):
        solution = np.full_like(rhs, np.nan)
    return solution, "cg_jacobi", iterations, relative_residual


def voxel_downsample_xyz(points: np.ndarray, leaf: float) -> np.ndarray:
    if leaf <= 0.0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / float(leaf)).astype(np.int64)
    coords = np.ascontiguousarray(coords)
    keys = coords.view(np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])).reshape(-1)
    _, inverse = np.unique(keys, return_inverse=True)
    count = int(inverse.max()) + 1 if inverse.size else 0
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    out = np.empty((count, 3), dtype=np.float64)
    for column in range(3):
        out[:, column] = np.bincount(inverse, weights=points[:, column], minlength=count) / counts
    return out


def prepare_local_cloud(points_xyz: np.ndarray, params: BalmParams) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if params.max_range > 0.0:
        points = points[np.einsum("ij,ij->i", points, points) <= params.max_range**2]
    return voxel_downsample_xyz(points, params.downsample_leaf)


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    axis = omega / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _group_stats(points: np.ndarray, group_ids: np.ndarray, n_groups: int):
    """Vectorized per-group count / mean / covariance via bincount."""
    counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    means = np.empty((n_groups, 3))
    for column in range(3):
        means[:, column] = (
            np.bincount(group_ids, weights=points[:, column], minlength=n_groups) / safe
        )
    covs = np.empty((n_groups, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            second = (
                np.bincount(
                    group_ids, weights=points[:, a] * points[:, b], minlength=n_groups
                )
                / safe
            )
            covs[:, a, b] = covs[:, b, a] = second - means[:, a] * means[:, b]
    return counts, means, covs


def extract_planar_voxels(
    world_points: np.ndarray,
    pose_ids: np.ndarray,
    params: BalmParams,
    sensor_xyz: np.ndarray | None = None,
    double_sided_diagnostics: list[dict] | None = None,
) -> List[np.ndarray]:
    """Level-by-level (BFS) fully vectorized adaptive voxel plane extraction.

    Equivalent to the recursive octree formulation: every layer halves the voxel
    size; each layer computes all voxel statistics in batched numpy calls.
    When sensor positions are supplied, opposite observed faces receive private
    branch identities and can never re-merge at a finer octree level.
    """
    total = world_points.shape[0]
    if total == 0:
        return []
    thickness_sq = params.plane_thickness**2
    spread_sq = (params.plane_spread_ratio * params.plane_thickness) ** 2

    # Layer 0: hash points into root voxels with a packed integer key so the
    # grouping sort is a single O(n) radix pass.
    voxel_size = params.root_voxel_size
    keys3 = np.floor(world_points / voxel_size).astype(np.int64)
    bias = np.int64(1) << 20
    packed = (
        ((keys3[:, 0] + bias).astype(np.uint64) << np.uint64(42))
        | ((keys3[:, 1] + bias).astype(np.uint64) << np.uint64(21))
        | (keys3[:, 2] + bias).astype(np.uint64)
    )
    order = np.argsort(packed, kind="stable")
    cur_rows = order
    packed = packed[order]
    boundaries = np.nonzero(np.diff(packed) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries, [total]])
    seg_len = np.diff(starts)
    n_groups = seg_len.size
    gid = np.repeat(np.arange(n_groups), seg_len)
    parent_center = (keys3[cur_rows[starts[:-1]]].astype(np.float64) + 0.5) * voxel_size

    unit_base = 0
    accepted_rows: List[np.ndarray] = []
    accepted_units: List[np.ndarray] = []

    points = world_points[cur_rows]
    for layer in range(params.max_layer + 1):
        if cur_rows.size < params.min_voxel_points:
            break
        big = seg_len >= params.min_voxel_points
        _, means, covs = _group_stats(points, gid, n_groups)
        eigenvalues, eigenvectors = np.linalg.eigh(covs)
        planar = big & (eigenvalues[:, 0] <= thickness_sq) & (eigenvalues[:, 1] >= spread_sq)

        is_double_sided = np.zeros(n_groups, dtype=bool)
        face_key = None
        face_planar = None
        if params.double_sided_enable and sensor_xyz is not None:
            normals_point = eigenvectors[:, :, 0][gid]
            view_dot = np.einsum(
                "ij,ij->i", sensor_xyz[cur_rows] - points, normals_point
            )
            face_sign = (view_dot >= 0.0).astype(np.int64)
            face_key = gid * 2 + face_sign
            face_counts = np.bincount(face_key, minlength=2 * n_groups)
            offsets = np.einsum(
                "ij,ij->i", points - means[gid], normals_point
            )
            face_sums = np.bincount(
                face_key, weights=offsets, minlength=2 * n_groups
            )
            counts_two = face_counts.reshape(-1, 2)
            both_supported = (
                counts_two >= params.min_voxel_points
            ).all(axis=1)
            face_means = face_sums.reshape(-1, 2) / np.maximum(counts_two, 1)
            face_separation = np.abs(face_means[:, 0] - face_means[:, 1])
            is_double_sided = (
                big & both_supported
                & (face_separation >= params.double_sided_min_separation)
            )
            if is_double_sided.any():
                subset = np.nonzero(is_double_sided[gid])[0]
                unique_faces, compact_face = np.unique(
                    face_key[subset], return_inverse=True
                )
                subset_counts = np.bincount(
                    compact_face, minlength=unique_faces.size
                )
                _, _, subset_covariances = _group_stats(
                    points[subset], compact_face, unique_faces.size
                )
                subset_eigenvalues = np.linalg.eigvalsh(subset_covariances)
                face_planar = np.zeros(2 * n_groups, dtype=bool)
                face_planar[unique_faces] = (
                    (subset_counts >= params.min_voxel_points)
                    & (subset_eigenvalues[:, 0] <= thickness_sq)
                    & (subset_eigenvalues[:, 1] >= spread_sq)
                )
                if double_sided_diagnostics is not None:
                    for group in np.nonzero(is_double_sided)[0]:
                        negative_ok = bool(face_planar[2 * group])
                        positive_ok = bool(face_planar[2 * group + 1])
                        if not (negative_ok and positive_ok):
                            continue
                        group_rows = gid == group
                        negative_rows = cur_rows[
                            group_rows & ((face_key & 1) == 0)
                        ]
                        positive_rows = cur_rows[
                            group_rows & ((face_key & 1) == 1)
                        ]
                        double_sided_diagnostics.append({
                            "layer": int(layer),
                            "center_world": means[group].copy(),
                            "normal_world": eigenvectors[group, :, 0].copy(),
                            "separation_m": float(face_separation[group]),
                            "negative_rows": negative_rows.copy(),
                            "positive_rows": positive_rows.copy(),
                            "negative_pose_count": int(np.unique(
                                pose_ids[negative_rows]
                            ).size),
                            "positive_pose_count": int(np.unique(
                                pose_ids[positive_rows]
                            ).size),
                        })

        point_is_double_sided = is_double_sided[gid]
        unit = np.full(cur_rows.size, -1, np.int64)
        combined_accept = planar[gid] & ~point_is_double_sided
        unit[combined_accept] = gid[combined_accept] * 3
        face_failed = np.zeros(cur_rows.size, dtype=bool)
        if face_planar is not None:
            face_ok = point_is_double_sided & face_planar[face_key]
            unit[face_ok] = gid[face_ok] * 3 + 1 + (face_key[face_ok] & 1)
            face_failed = point_is_double_sided & ~face_planar[face_key]

        keep_units = unit >= 0
        if keep_units.any():
            accepted_rows.append(cur_rows[keep_units])
            accepted_units.append(unit[keep_units] + unit_base)
        unit_base += 3 * n_groups

        if layer >= params.max_layer:
            break
        carry = (
            (big[gid] & ~planar[gid] & ~point_is_double_sided)
            | face_failed
        )
        if not carry.any():
            break
        carried_rows = cur_rows[carry]
        carried_gid = gid[carry]
        carried_pts = points[carry]
        octant = (
            (carried_pts[:, 0] > parent_center[carried_gid, 0]).astype(np.int64)
            | ((carried_pts[:, 1] > parent_center[carried_gid, 1]).astype(np.int64) << 1)
            | ((carried_pts[:, 2] > parent_center[carried_gid, 2]).astype(np.int64) << 2)
        )
        face_bit = np.zeros(carried_rows.size, dtype=np.int64)
        if face_key is not None:
            face_bit = (
                (face_key[carry] & 1) * face_failed[carry].astype(np.int64)
            )
        # The face bit is part of the branch identity. Once separated, two
        # physical faces cannot meet again in a finer child voxel.
        label = carried_gid * 16 + octant * 2 + face_bit
        order = np.argsort(label, kind="stable")
        carried_rows = carried_rows[order]
        carried_pts = carried_pts[order]
        label = label[order]
        boundaries = np.nonzero(np.diff(label) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries, [carried_rows.size]])
        seg_len = np.diff(starts)
        group_label = label[starts[:-1]]
        old_gid = group_label // 16
        group_octant = (group_label % 16) // 2
        quarter = voxel_size / 4.0
        parent_center = parent_center[old_gid] + np.column_stack(
            [
                np.where(group_octant & 1, quarter, -quarter),
                np.where(group_octant & 2, quarter, -quarter),
                np.where(group_octant & 4, quarter, -quarter),
            ]
        )
        n_groups = seg_len.size
        gid = np.repeat(np.arange(n_groups), seg_len)
        cur_rows = carried_rows
        points = carried_pts
        voxel_size /= 2.0

    if not accepted_rows:
        return []
    rows_all = np.concatenate(accepted_rows)
    units_all = np.concatenate(accepted_units)
    _, unit_idx = np.unique(units_all, return_inverse=True)
    n_units = int(unit_idx.max()) + 1
    poses = pose_ids[rows_all]
    pose_span = np.int64(pose_ids.max()) + 1
    order = np.argsort(unit_idx.astype(np.int64) * pose_span + poses, kind="stable")
    rows_all = rows_all[order]
    unit_idx = unit_idx[order]
    poses = poses[order]

    pair_change = np.empty(rows_all.size, dtype=bool)
    pair_change[0] = True
    pair_change[1:] = (unit_idx[1:] != unit_idx[:-1]) | (poses[1:] != poses[:-1])
    pose_count_per_unit = np.bincount(unit_idx[pair_change], minlength=n_units)
    unit_ok = pose_count_per_unit >= params.min_poses_per_voxel

    keep = unit_ok[unit_idx]
    cap = params.max_points_per_voxel_pose
    if cap > 0:
        pair_id = np.cumsum(pair_change) - 1
        seg_starts = np.nonzero(pair_change)[0]
        rank = np.arange(rows_all.size) - seg_starts[pair_id]
        seg_sizes = np.diff(np.append(seg_starts, rows_all.size))
        seg_of_point = seg_sizes[pair_id]
        keep &= ((rank + 1) * cap) // seg_of_point > (rank * cap) // seg_of_point
    rows_keep = rows_all[keep]
    unit_keep = unit_idx[keep]
    if rows_keep.size == 0:
        return []
    split_at = np.nonzero(np.diff(unit_keep))[0] + 1
    return list(np.split(rows_keep, split_at))


def _plane_geometry(
    world: np.ndarray,
    plane_ids: np.ndarray,
    plane_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the nuisance plane of every factor and return its residuals."""
    plane_sizes = np.bincount(plane_ids, minlength=plane_count).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(plane_ids, weights=world[:, column], minlength=plane_count)
            for column in range(3)
        ],
        axis=1,
    )
    centroids = sums / plane_sizes[:, None]
    moments = np.empty((plane_count, 3, 3))
    for row in range(3):
        for column in range(row, 3):
            value = (
                np.bincount(
                    plane_ids,
                    weights=world[:, row] * world[:, column],
                    minlength=plane_count,
                )
                / plane_sizes
            )
            moments[:, row, column] = value
            moments[:, column, row] = value
    covariances = moments - centroids[:, :, None] * centroids[:, None, :]
    _, eigenvectors = np.linalg.eigh(covariances)
    normals = eigenvectors[:, :, 0]
    residuals = np.einsum(
        "mi,mi->m", world - centroids[plane_ids], normals[plane_ids]
    )
    return centroids, eigenvectors, normals, residuals


def _robust_weights_and_cost(
    residuals: np.ndarray,
    params: BalmParams,
) -> tuple[np.ndarray, float]:
    """Return IRLS weights and the matching robust objective."""
    if params.robust_kernel == "none":
        return np.ones_like(residuals), float(0.5 * np.dot(residuals, residuals))
    delta = max(1e-6, params.robust_delta_scale * params.plane_thickness)
    absolute = np.abs(residuals)
    if params.robust_kernel == "cauchy":
        scaled = residuals / delta
        weights = 1.0 / (1.0 + scaled**2)
        cost = 0.5 * delta**2 * np.log1p(scaled**2).sum()
    else:
        weights = np.minimum(1.0, delta / np.maximum(absolute, 1e-12))
        quadratic = absolute <= delta
        cost = (
            0.5 * np.square(residuals[quadratic]).sum()
            + (delta * (absolute[~quadratic] - 0.5 * delta)).sum()
        )
    return weights, float(cost)


def _pose_prior_terms(
    rotations: np.ndarray,
    translations: np.ndarray,
    reference_rotations: np.ndarray,
    reference_translations: np.ndarray,
    params: BalmParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diagonal pose-graph trust term in each pose's body tangent."""
    variable_count = max(0, rotations.shape[0] - 1) * 6
    diagonal = np.zeros(variable_count, dtype=np.float64)
    gradient = np.zeros(variable_count, dtype=np.float64)
    objective = 0.0
    weights = np.asarray(
        [params.pose_prior_rotation_weight] * 3
        + [params.pose_prior_translation_weight] * 3,
        dtype=np.float64,
    )
    if not np.any(weights > 0.0):
        return diagonal, gradient, objective
    for index in range(1, rotations.shape[0]):
        rotation_error = Rotation.from_matrix(
            reference_rotations[index].T @ rotations[index]
        ).as_rotvec()
        translation_error = rotations[index].T @ (
            translations[index] - reference_translations[index]
        )
        error = np.concatenate([rotation_error, translation_error])
        span = slice((index - 1) * 6, index * 6)
        diagonal[span] = weights
        gradient[span] = weights * error
        objective += 0.5 * float(np.dot(weights, error**2))
    return diagonal, gradient, objective


def _joint_plane_system(
    local_points: np.ndarray,
    pose_ids: np.ndarray,
    plane_ids: np.ndarray,
    world: np.ndarray,
    rotations: np.ndarray,
    centroids: np.ndarray,
    eigenvectors: np.ndarray,
    normals: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build the Schur-reduced joint pose system.

    Each plane contributes two normal directions and one offset as nuisance
    variables. Eliminating those three variables creates the inter-pose blocks
    missing from independent per-keyframe point-to-plane solves.
    """
    point_count = local_points.shape[0]
    pose_count = rotations.shape[0]
    plane_count = centroids.shape[0]
    variable_count = max(0, pose_count - 1) * 6

    body_normals = np.einsum(
        "mi,mij->mj", normals[plane_ids], rotations[pose_ids]
    )
    jacobians = np.empty((point_count, 6), dtype=np.float64)
    jacobians[:, :3] = np.cross(local_points, body_normals)
    jacobians[:, 3:] = body_normals

    centered = world - centroids[plane_ids]
    tangent_one = eigenvectors[plane_ids, :, 1]
    tangent_two = eigenvectors[plane_ids, :, 2]
    feature_jacobians = np.column_stack(
        [
            np.einsum("mi,mi->m", centered, tangent_one),
            np.einsum("mi,mi->m", centered, tangent_two),
            np.ones(point_count, dtype=np.float64),
        ]
    )
    sqrt_weights = np.sqrt(weights)
    jacobians *= sqrt_weights[:, None]
    feature_jacobians *= sqrt_weights[:, None]
    weighted_residuals = residuals * sqrt_weights

    movable = pose_ids > 0
    movable_rows = np.nonzero(movable)[0]
    pose_columns = (pose_ids[movable] - 1) * 6
    pose_jacobian = sparse.csr_matrix(
        (
            jacobians[movable].reshape(-1),
            (
                np.repeat(movable_rows, 6),
                (pose_columns[:, None] + np.arange(6)).reshape(-1),
            ),
        ),
        shape=(point_count, variable_count),
    )
    feature_jacobian = sparse.csr_matrix(
        (
            feature_jacobians.reshape(-1),
            (
                np.repeat(np.arange(point_count), 3),
                (plane_ids[:, None] * 3 + np.arange(3)).reshape(-1),
            ),
        ),
        shape=(point_count, plane_count * 3),
    )

    pose_hessian = (pose_jacobian.T @ pose_jacobian).tocsr()
    cross_hessian = (pose_jacobian.T @ feature_jacobian).tocsr()
    feature_blocks = np.zeros((plane_count, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(row, 3):
            values = np.bincount(
                plane_ids,
                weights=feature_jacobians[:, row] * feature_jacobians[:, column],
                minlength=plane_count,
            )
            feature_blocks[:, row, column] = values
            feature_blocks[:, column, row] = values
    scale = np.maximum(np.trace(feature_blocks, axis1=1, axis2=2), 1.0)
    feature_blocks += (1e-9 * scale)[:, None, None] * np.eye(3)[None, :, :]
    inverse_blocks = np.linalg.inv(feature_blocks)
    inverse_feature_hessian = sparse.block_diag(inverse_blocks, format="csr")

    feature_gradient = np.asarray(
        feature_jacobian.T @ weighted_residuals
    ).reshape(-1)
    pose_gradient = np.asarray(pose_jacobian.T @ weighted_residuals).reshape(-1)
    reduced_hessian = (
        pose_hessian
        - cross_hessian @ inverse_feature_hessian @ cross_hessian.T
    ).tocsr()
    reduced_hessian = ((reduced_hessian + reduced_hessian.T) * 0.5).tocsr()
    reduced_gradient = pose_gradient - np.asarray(
        cross_hessian @ (inverse_feature_hessian @ feature_gradient)
    ).reshape(-1)
    return reduced_hessian, reduced_gradient


def _transform_selected_points(
    local_points: np.ndarray,
    pose_ids: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    return np.einsum("mi,mji->mj", local_points, rotations[pose_ids]) + translations[pose_ids]


def run_balm_refinement(
    local_clouds: List[np.ndarray],
    poses_world_sensor: np.ndarray,
    params: BalmParams,
    log_fn: LogFn = None,
    reference_poses_world_sensor: np.ndarray | None = None,
) -> BalmResult:
    """Jointly refine keyframe poses after eliminating each fitted plane.

    ``local_clouds[i]`` contains sensor-frame xyz points for keyframe ``i``.
    Pose zero is fixed as the gauge anchor. Every candidate LM step is evaluated
    on the same point-to-plane associations and rolled back if it does not lower
    the robust objective.
    """
    started = time.perf_counter()
    poses_world_sensor = np.asarray(poses_world_sensor, dtype=np.float64)
    pose_count = len(local_clouds)
    if poses_world_sensor.shape != (pose_count, 4, 4):
        raise ValueError(
            f"Pose shape {poses_world_sensor.shape} does not match {pose_count} clouds."
        )
    rotations = poses_world_sensor[:, :3, :3].copy()
    translations = poses_world_sensor[:, :3, 3].copy()
    if reference_poses_world_sensor is None:
        reference_poses_world_sensor = poses_world_sensor
    reference_poses_world_sensor = np.asarray(
        reference_poses_world_sensor, dtype=np.float64
    )
    if reference_poses_world_sensor.shape != poses_world_sensor.shape:
        raise ValueError("reference_poses_world_sensor must match poses_world_sensor shape")
    reference_rotations = reference_poses_world_sensor[:, :3, :3].copy()
    reference_translations = reference_poses_world_sensor[:, :3, 3].copy()

    counts = [cloud.shape[0] for cloud in local_clouds]
    all_local = (
        np.vstack([cloud for cloud in local_clouds if cloud.shape[0]])
        if any(counts)
        else np.empty((0, 3), dtype=np.float64)
    )
    all_pose = np.repeat(np.arange(pose_count), counts)
    result = BalmResult(poses_world_sensor=poses_world_sensor.copy())
    if all_local.shape[0] == 0 or pose_count < 2:
        result.stop_reason = "insufficient_data"
        _log(log_fn, "[BALM] Insufficient points or poses; skipping refinement.")
        result.elapsed_sec = time.perf_counter() - started
        return result

    sel_local: Optional[np.ndarray] = None
    sel_pose: Optional[np.ndarray] = None
    sel_plane: Optional[np.ndarray] = None
    plane_count = 0
    rms_history: List[float] = []
    motion_since_association = float("inf")
    lm_damping = max(float(params.lm_damping), 1e-9)
    double_sided_voxel_count = 0

    for iteration in range(params.max_iterations):
        reassociated = False
        association_due = (
            params.reassociate_every > 0
            and iteration % params.reassociate_every == 0
            and motion_since_association > params.reassociate_min_motion
        )
        if sel_local is None or association_due:
            world_all = _transform_selected_points(
                all_local, all_pose, rotations, translations
            )
            double_sided_diagnostics = (
                [] if params.double_sided_enable else None
            )
            plane_rows = extract_planar_voxels(
                world_all,
                all_pose,
                params,
                translations[all_pose],
                double_sided_diagnostics,
            )
            double_sided_voxel_count = (
                len(double_sided_diagnostics)
                if double_sided_diagnostics is not None else 0
            )
            if not plane_rows:
                result.stop_reason = "no_planes"
                _log(log_fn, "[BALM] No planar voxels found; nothing to refine.")
                break
            sel_rows = np.concatenate(plane_rows)
            sel_plane = np.repeat(
                np.arange(len(plane_rows)), [rows.size for rows in plane_rows]
            )
            sel_local = all_local[sel_rows]
            sel_pose = all_pose[sel_rows]
            plane_count = len(plane_rows)
            reassociated = True
            motion_since_association = 0.0

            # Re-association changes the residual set and therefore the local
            # quadratic model.  Carrying a tiny damping value learned on the
            # previous model lets the first step on the new model explode
            # along weak plane directions.  Keep any damping that was raised
            # after a rejection, but never carry an aggressively decayed
            # trust region across model changes.
            lm_damping = max(lm_damping, float(params.lm_damping))

        assert sel_local is not None and sel_pose is not None and sel_plane is not None
        point_count = sel_local.shape[0]
        world = _transform_selected_points(
            sel_local, sel_pose, rotations, translations
        )
        centroids, eigenvectors, normals, residuals = _plane_geometry(
            world, sel_plane, plane_count
        )
        weights, plane_objective = _robust_weights_and_cost(residuals, params)
        prior_diagonal, prior_gradient, prior_objective = _pose_prior_terms(
            rotations,
            translations,
            reference_rotations,
            reference_translations,
            params,
        )
        current_objective = plane_objective + prior_objective
        hessian, gradient = _joint_plane_system(
            sel_local,
            sel_pose,
            sel_plane,
            world,
            rotations,
            centroids,
            eigenvectors,
            normals,
            residuals,
            weights,
        )
        if np.any(prior_diagonal > 0.0):
            hessian = hessian + sparse.diags(prior_diagonal, format="csr")
            gradient = gradient + prior_gradient

        hessian_diagonal = np.maximum(np.abs(hessian.diagonal()), 1e-9)
        accepted = False
        accepted_rms = float(np.sqrt(np.mean(residuals**2)))
        accepted_objective = current_objective
        accepted_rotations = rotations
        accepted_translations = translations
        accepted_update_norms = np.zeros(pose_count, dtype=np.float64)
        accepted_rotation_update_norms = np.zeros(
            pose_count, dtype=np.float64
        )
        accepted_translation_update_norms = np.zeros(
            pose_count, dtype=np.float64
        )
        used_damping = lm_damping
        used_linear_solver = ""
        used_linear_iterations = 0
        used_linear_residual = float("inf")

        variable_count = int(hessian.shape[0])
        preferred_solver = (
            "sparse_direct"
            if variable_count <= _DIRECT_SOLVER_MAX_VARIABLES
            else "cg_jacobi"
        )
        _log(
            log_fn,
            f"[BALM] linear system variables={variable_count} "
            f"nnz={hessian.nnz} backend={preferred_solver}",
        )

        # Six trials cover a factor of 1e5 in damping without exposing another
        # tuning parameter. A rejected trial never changes the trajectory.
        for _ in range(6):
            damped = hessian + sparse.diags(
                used_damping * hessian_diagonal + 1e-9, format="csr"
            )
            (delta, used_linear_solver, used_linear_iterations,
             used_linear_residual) = _solve_damped_system(damped, -gradient)
            if not np.all(np.isfinite(delta)):
                _log(
                    log_fn,
                    f"[BALM] {used_linear_solver} did not converge "
                    f"(iterations={used_linear_iterations}, "
                    f"relative_residual={used_linear_residual:.3g}); "
                    "increasing LM damping",
                )
                used_damping *= 10.0
                continue
            candidate_rotations = rotations.copy()
            candidate_translations = translations.copy()
            update_norms = np.zeros(pose_count, dtype=np.float64)
            rotation_update_norms = np.zeros(pose_count, dtype=np.float64)
            translation_update_norms = np.zeros(pose_count, dtype=np.float64)
            for index in range(1, pose_count):
                step = delta[(index - 1) * 6:index * 6]
                rot_step = step[:3]
                trans_step = step[3:]
                rot_norm = float(np.linalg.norm(rot_step))
                trans_norm = float(np.linalg.norm(trans_step))
                if rot_norm > params.max_rot_step_rad:
                    rot_step *= params.max_rot_step_rad / rot_norm
                    rot_norm = params.max_rot_step_rad
                if trans_norm > params.max_trans_step_m:
                    trans_step *= params.max_trans_step_m / trans_norm
                    trans_norm = params.max_trans_step_m
                old_rotation = rotations[index]
                candidate_translations[index] += old_rotation @ trans_step
                candidate_rotations[index] = old_rotation @ _so3_exp(rot_step)
                update_norms[index] = max(rot_norm, trans_norm)
                rotation_update_norms[index] = rot_norm
                translation_update_norms[index] = trans_norm
            candidate_rotations = Rotation.from_matrix(candidate_rotations).as_matrix()
            candidate_world = _transform_selected_points(
                sel_local, sel_pose, candidate_rotations, candidate_translations
            )
            _, _, _, candidate_residuals = _plane_geometry(
                candidate_world, sel_plane, plane_count
            )
            _, candidate_plane_objective = _robust_weights_and_cost(
                candidate_residuals, params
            )
            _, _, candidate_prior_objective = _pose_prior_terms(
                candidate_rotations,
                candidate_translations,
                reference_rotations,
                reference_translations,
                params,
            )
            candidate_objective = candidate_plane_objective + candidate_prior_objective
            if candidate_objective < current_objective - 1e-12:
                accepted = True
                accepted_objective = candidate_objective
                accepted_rms = float(np.sqrt(np.mean(candidate_residuals**2)))
                accepted_rotations = candidate_rotations
                accepted_translations = candidate_translations
                accepted_update_norms = update_norms
                accepted_rotation_update_norms = rotation_update_norms
                accepted_translation_update_norms = translation_update_norms
                break
            used_damping *= 10.0

        max_update = float(accepted_update_norms.max()) if accepted else 0.0
        max_translation_update_m = (
            float(accepted_translation_update_norms.max()) if accepted else 0.0
        )
        p95_translation_update_m = (
            float(np.percentile(accepted_translation_update_norms[1:], 95))
            if accepted and pose_count > 1 else 0.0
        )
        max_rotation_update_rad = (
            float(accepted_rotation_update_norms.max()) if accepted else 0.0
        )
        p95_rotation_update_rad = (
            float(np.percentile(accepted_rotation_update_norms[1:], 95))
            if accepted and pose_count > 1 else 0.0
        )
        max_rotation_update_deg = float(np.degrees(max_rotation_update_rad))
        p95_rotation_update_deg = float(np.degrees(p95_rotation_update_rad))
        if accepted:
            rotations = accepted_rotations
            translations = accepted_translations
            typical_motion = float(np.median(accepted_update_norms[1:]))
            motion_since_association += typical_motion
            lm_damping = max(used_damping * 0.3, 1e-9)
        else:
            lm_damping = used_damping

        stats = BalmIterationStats(
            iteration=iteration,
            plane_count=plane_count,
            point_count=point_count,
            rms_plane_distance=accepted_rms,
            max_pose_update=max_update,
            reassociated=reassociated,
            max_translation_update_m=max_translation_update_m,
            p95_translation_update_m=p95_translation_update_m,
            max_rotation_update_deg=max_rotation_update_deg,
            p95_rotation_update_deg=p95_rotation_update_deg,
            objective=accepted_objective,
            accepted=accepted,
            lm_damping=used_damping,
            double_sided_voxel_count=double_sided_voxel_count,
            linear_solver=used_linear_solver,
            linear_solver_iterations=used_linear_iterations,
            linear_solver_relative_residual=used_linear_residual,
        )
        result.iterations.append(stats)
        _log(
            log_fn,
            f"[BALM] iter={iteration} planes={plane_count} points={point_count} "
            f"rms={accepted_rms:.4f} m objective={accepted_objective:.6g} "
            f"max_trans={max_translation_update_m:.5f} m "
            f"max_rot={max_rotation_update_deg:.5f} deg "
            f"p95_trans={p95_translation_update_m:.5f} m "
            f"p95_rot={p95_rotation_update_deg:.5f} deg "
            f"max_update={max_update:.5f} damping={used_damping:.2g} "
            f"linear_solver={used_linear_solver} "
            f"linear_iterations={used_linear_iterations} "
            f"linear_residual={used_linear_residual:.3g} "
            f"double_sided={double_sided_voxel_count} "
            f"{'accepted' if accepted else 'rejected'}"
            + (" (reassociated)" if reassociated else ""),
        )
        if not accepted:
            # A failed descent search can mean a stationary point, a poor
            # linear solve, or an invalid local model.  It is a legitimate
            # stop condition, but not proof that pose updates converged.
            result.converged = False
            result.stop_reason = "no_descent_step"
            break
        if (
            max_translation_update_m < params.convergence_tol
            and max_rotation_update_rad < params.convergence_tol
        ):
            result.converged = True
            result.stop_reason = "pose_converged"
            break
        rms_history.append(accepted_rms)
        if _rms_has_plateaued(
            rms_history,
            params.plateau_window,
            params.plateau_min_improvement,
        ):
            # Re-associated RMS values are a useful deployment-time plateau
            # heuristic, but each value can describe a different residual set.
            # Do not promote this practical stop to numerical pose convergence.
            result.converged = False
            result.stop_reason = "objective_plateau"
            _log(
                log_fn,
                f"[BALM] Early stop: rms plateaued over the last "
                f"{params.plateau_window} iterations.",
            )
            break

    poses_out = np.tile(np.eye(4), (pose_count, 1, 1))
    poses_out[:, :3, :3] = rotations
    poses_out[:, :3, 3] = translations
    result.poses_world_sensor = poses_out
    result.elapsed_sec = time.perf_counter() - started
    return result


def _contiguous_keyframe_ranges(indices: np.ndarray, max_gap: int) -> List[List[int]]:
    ranges: List[List[int]] = []
    for index in np.sort(indices):
        index = int(index)
        if ranges and index - ranges[-1][1] <= max_gap:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ranges


def _two_means_1d(values: np.ndarray, iterations: int = 25):
    center_low, center_high = float(values.min()), float(values.max())
    low_mask = values <= (center_low + center_high) / 2.0
    for _ in range(iterations):
        if not low_mask.any() or low_mask.all():
            return None
        new_low = float(values[low_mask].mean())
        new_high = float(values[~low_mask].mean())
        if abs(new_low - center_low) < 1e-9 and abs(new_high - center_high) < 1e-9:
            break
        center_low, center_high = new_low, new_high
        low_mask = values <= (center_low + center_high) / 2.0
    return center_low, center_high, low_mask


def _canonical_plane_normal(normal: np.ndarray) -> np.ndarray:
    """Give an unoriented plane normal a deterministic sign."""
    normal = np.asarray(normal, dtype=np.float64)
    axis = int(np.argmax(np.abs(normal)))
    return -normal if normal[axis] < 0.0 else normal


def _in_plane_overlap_metrics(
    points: np.ndarray,
    low_mask: np.ndarray,
    tangent_basis: np.ndarray,
    bin_size: float,
) -> tuple[float, int, float]:
    """Return ratio, cell count and area of projected two-layer overlap.

    A pose ghost duplicates the *same surface support*.  Corners, door edges and
    unrelated structures can be bimodal along a PCA normal, but their projected
    support usually does not overlap.  The denominator is the smaller layer so
    partial visibility through an opening remains admissible.  Physical overlap
    area is retained as the density-invariant ranking signal; point count is not
    a stable severity measure across sensors or vehicle speeds.
    """
    if not low_mask.any() or low_mask.all():
        return 0.0, 0, 0.0
    bin_edge = max(float(bin_size), 1e-6)
    projected = points @ tangent_basis
    cells = np.floor(projected / bin_edge).astype(np.int64)
    dtype = np.dtype([("u", np.int64), ("v", np.int64)])

    def occupied(mask: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(cells[mask])
        return np.unique(contiguous.view(dtype).reshape(-1))

    occupied_low = occupied(low_mask)
    occupied_high = occupied(~low_mask)
    denominator = min(occupied_low.size, occupied_high.size)
    if denominator == 0:
        return 0.0, 0, 0.0
    overlap_cells = int(np.intersect1d(occupied_low, occupied_high).size)
    return (
        float(overlap_cells / denominator),
        overlap_cells,
        float(overlap_cells * bin_edge**2),
    )


def _in_plane_overlap_ratio(
    points: np.ndarray,
    low_mask: np.ndarray,
    tangent_basis: np.ndarray,
    bin_size: float,
) -> float:
    """Compatibility wrapper returning only the projected occupancy ratio."""
    return _in_plane_overlap_metrics(
        points, low_mask, tangent_basis, bin_size
    )[0]


def _interval_distance(interval_a: list[int], interval_b: list[int]) -> int:
    """Number of keyframes separating two closed intervals (zero if overlapping)."""
    if interval_a[1] < interval_b[0]:
        return int(interval_b[0] - interval_a[1])
    if interval_b[1] < interval_a[0]:
        return int(interval_a[0] - interval_b[1])
    return 0


def _traversal_pair_distance(region_a: dict, region_b: dict) -> int:
    """Distance between unordered pairs of traversal intervals.

    The most populated keyframe is unstable between adjacent surface patches;
    the traversal interval is the physically meaningful identity of a layer.
    """
    a0, a1 = region_a["layer_a_keyframes"], region_a["layer_b_keyframes"]
    b0, b1 = region_b["layer_a_keyframes"], region_b["layer_b_keyframes"]
    direct = max(_interval_distance(a0, b0), _interval_distance(a1, b1))
    crossed = max(_interval_distance(a0, b1), _interval_distance(a1, b0))
    return int(min(direct, crossed))


def _merge_connected_ghost_voxels(
    raw_regions: List[dict],
    *,
    max_kf_gap: int,
    min_region_voxels: int,
    normal_cos_min: float,
    separation_tolerance: float,
    max_spatial_voxel_gap: int = 2,
    coplanar_tolerance: float = 0.35,
) -> List[dict]:
    """Merge spatially coherent ghost patches, tolerating one missing voxel.

    Occlusion and per-keyframe downsampling can leave a one-voxel hole in an
    otherwise visible duplicated surface.  Geometry and traversal identity are
    still required, so this tolerance does not reconnect unrelated candidates.
    """
    if not raw_regions:
        return []

    adjacency: list[list[int]] = [[] for _ in raw_regions]
    for left in range(len(raw_regions)):
        region_left = raw_regions[left]
        key_left = np.asarray(region_left["voxel_key"], dtype=np.int64)
        normal_left = np.asarray(region_left["normal"], dtype=np.float64)
        for right in range(left + 1, len(raw_regions)):
            region_right = raw_regions[right]
            if region_left["view_side"] != region_right["view_side"]:
                continue
            key_right = np.asarray(region_right["voxel_key"], dtype=np.int64)
            # Permit one empty root voxel between coherent surface patches.
            if np.max(np.abs(key_left - key_right)) > max_spatial_voxel_gap:
                continue
            if _traversal_pair_distance(region_left, region_right) > max_kf_gap:
                continue
            normal_right = np.asarray(region_right["normal"], dtype=np.float64)
            if abs(float(normal_left @ normal_right)) < normal_cos_min:
                continue
            center_delta = (
                np.asarray(region_right["center_xyz"], dtype=np.float64)
                - np.asarray(region_left["center_xyz"], dtype=np.float64)
            )
            # Nearby parallel structures are not one duplicated surface.  Their
            # centers must agree in the normal direction and extend tangentially.
            if abs(float(center_delta @ normal_left)) > coplanar_tolerance:
                continue
            if abs(
                float(region_left["separation_m"])
                - float(region_right["separation_m"])
            ) > separation_tolerance:
                continue
            adjacency[left].append(right)
            adjacency[right].append(left)

    components: list[list[int]] = []
    unseen = set(range(len(raw_regions)))
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        if len(component) >= min_region_voxels:
            components.append(component)

    merged: List[dict] = []
    for component in components:
        members = [raw_regions[index] for index in component]
        point_weights = np.asarray(
            [member["point_count"] for member in members], dtype=float
        )
        overlap_areas = np.asarray(
            [member.get("overlap_area_m2", 0.0) for member in members],
            dtype=float,
        )
        # Old synthetic callers may not carry physical overlap yet. Preserve
        # their behavior, while detector-produced candidates use area weights.
        weights = (
            overlap_areas if float(overlap_areas.sum()) > 0.0
            else point_weights
        )
        separations = np.asarray([member["separation_m"] for member in members], dtype=float)
        max_member = members[int(np.argmax(separations))]
        representative = max(
            members,
            key=lambda item: (
                item.get("overlap_area_m2", 0.0) * item["separation_m"],
                item["point_count"] * item["separation_m"],
            ),
        )
        normals = np.asarray([member["normal"] for member in members], dtype=float)
        reference_normal = normals[0]
        normals[normals @ reference_normal < 0.0] *= -1.0
        mean_normal = np.average(normals, axis=0, weights=weights)
        mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
        centroid = np.average(
            np.asarray([member["center_xyz"] for member in members], dtype=float),
            axis=0,
            weights=weights,
        )
        merged.append(
            {
                # Legacy consumers pair center_xyz with separation_m.  Keep both
                # from the same raw voxel; expose the component centroid separately.
                "center_xyz": max_member["center_xyz"],
                "region_centroid_xyz": [round(float(value), 2) for value in centroid],
                "separation_m": round(float(separations.max()), 3),
                "separation_typical_m": round(
                    float(np.average(separations, weights=weights)), 3
                ),
                "normal": [round(float(value), 4) for value in mean_normal],
                "point_count": int(point_weights.sum()),
                "overlap_area_m2": round(float(overlap_areas.sum()), 3),
                "surface_displacement_m3": round(
                    float(np.sum(overlap_areas * separations)), 5
                ),
                "asymmetric": all(member["asymmetric"] for member in members),
                "voxel_count": len(members),
                "voxel_keys": [member["voxel_key"] for member in members],
                # Preserve the per-patch descriptors needed to reproduce the
                # selection. Exact point identities are deliberately kept out
                # of JSON; the legacy causal-audit path reselects them and
                # rejects the evidence if separation differs by more than 1 cm.
                "component_voxels": [
                    {
                        "voxel_key": member["voxel_key"],
                        "view_side": member["view_side"],
                        "normal": member["normal"],
                        "separation_m": member["separation_m"],
                        "point_count": member["point_count"],
                        "layer_a_keyframes": member["layer_a_keyframes"],
                        "layer_b_keyframes": member["layer_b_keyframes"],
                    }
                    for member in members
                ],
                "layer_overlap_ratio": round(
                    float(np.average(
                        [member["layer_overlap_ratio"] for member in members],
                        weights=weights,
                    )),
                    3,
                ),
                "layer_support_fraction": round(
                    float(min(member["layer_support_fraction"] for member in members)),
                    3,
                ),
                "same_side_observation": True,
                # A coherent duplicated plane is a map-quality observation, not
                # a six-DoF loop certificate. A single normal leaves tangent
                # motion and rotation underconstrained; GICP/PGO must establish
                # those separately in the explicit legacy proposal workflow.
                "candidate_type": "duplicated_surface_region",
                "loop_observability": "single_plane_underconstrained",
                "loop_constraint_ready": False,
                "layer_a_keyframes": representative["layer_a_keyframes"],
                "layer_b_keyframes": representative["layer_b_keyframes"],
                "suggested_pair": representative["suggested_pair"],
            }
        )
    merged.sort(
        key=lambda region: (
            -region["surface_displacement_m3"],
            -region["separation_typical_m"],
        )
    )
    return merged


def detect_ghost_regions(
    local_clouds: List[np.ndarray],
    poses_world_sensor: np.ndarray,
    params: BalmParams,
    *,
    min_separation: float = 0.08,
    max_separation: float = 0.35,
    layer_std_max: float = 0.05,
    min_layer_fraction: float = 0.2,
    min_minor_layer_points: int = 20,
    min_minor_layer_keyframes: int = 3,
    min_layer_keyframes: int = 3,
    min_in_plane_overlap: float = 0.35,
    overlap_bin_size: float = 0.25,
    min_traversal_purity: float = 0.40,
    allow_asymmetric_layers: bool = False,
    min_region_voxels: int = 2,
    region_normal_cos_min: float = 0.965,
    region_separation_tolerance: float = 0.06,
    max_spatial_voxel_gap: int = 2,
    region_coplanar_tolerance: float = 0.35,
    max_kf_gap: int = 20,
    max_regions: int = 10,
    max_observation_range: float = 20.0,
    log_fn: LogFn = None,
) -> List[dict]:
    """Scan the refined map for residual duplicated planar surfaces.

    A single bimodal voxel is only a *double-layer candidate*.  It is promoted
    to a duplicated-surface region only when both layers (1) are observed from the same side,
    (2) overlap after projection onto the surface plane, (3) are supported by
    two temporally separated, reasonably pure traversals, and (4) persist as a
    spatially connected component whose normals and separation agree.  These
    conditions reject real opposite wall faces, corners, door edges, isolated
    outliers, and the former failure mode that merged voxels many metres apart.
    The result diagnoses map quality; a single planar component does not make
    a six-DoF loop constraint observable.

    ``center_xyz`` and ``separation_m`` refer to the same maximum-separation raw
    voxel. ``region_centroid_xyz`` and ``separation_typical_m`` summarize the
    connected component and are appropriate for localization and comparison.
    """
    pose_count = len(local_clouds)
    counts = [cloud.shape[0] for cloud in local_clouds]
    if not any(counts):
        return []
    rotations = poses_world_sensor[:, :3, :3]
    translations = poses_world_sensor[:, :3, 3]
    all_pose = np.repeat(np.arange(pose_count), counts)
    world = np.empty((int(np.sum(counts)), 3))
    cursor = 0
    for index in range(pose_count):
        stop = cursor + counts[index]
        if stop > cursor:
            world[cursor:stop] = (
                local_clouds[index] @ rotations[index].T + translations[index]
            )
        cursor = stop
    sensor_xyz = translations[all_pose]

    # Pose-error ghosting is range-independent, but angular-noise misplacement
    # grows linearly with range: distant observations of a surface masquerade
    # as ghost layers. Restrict the diagnosis to nearby observations.
    if max_observation_range > 0.0:
        near = (
            np.einsum("ij,ij->i", world - sensor_xyz, world - sensor_xyz)
            <= max_observation_range**2
        )
        world = world[near]
        all_pose = all_pose[near]
        sensor_xyz = sensor_xyz[near]

    keys = np.floor(world / params.root_voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.nonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1))[0] + 1
    starts = np.concatenate([[0], boundaries, [order.size]])

    min_points = 2 * params.min_voxel_points
    raw_regions: List[dict] = []
    for begin, end in zip(starts[:-1], starts[1:]):
        rows = order[begin:end]
        if rows.size < min_points:
            continue
        points = world[rows]
        mean = points.mean(axis=0)
        centered = points - mean
        cov = centered.T @ centered / rows.size
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if eigenvalues[1] < 0.01:  # no meaningful in-plane extent
            continue
        normal = _canonical_plane_normal(eigenvectors[:, 0])
        tangent_basis = eigenvectors[:, 1:3]
        # Opposite faces of a real wall have opposite sensor-to-surface signs.
        # Diagnose each observing side independently.
        view_dot = np.einsum("ij,j->i", sensor_xyz[rows] - points, normal)
        for view_side, side_mask in ((1, view_dot >= 0.0), (-1, view_dot < 0.0)):
            side_rows = rows[side_mask]
            if side_rows.size < min_points:
                continue
            side_points = world[side_rows]
            side_offsets = (side_points - mean) @ normal
            if (
                side_offsets.std() < min_separation / 2.0
                and np.percentile(side_offsets, 99.5)
                - np.percentile(side_offsets, 0.5)
                < min_separation
            ):
                continue
            split = _two_means_1d(side_offsets)
            if split is None:
                continue
            center_low, center_high, low_mask = split
            separation = center_high - center_low
            fraction = float(low_mask.mean())
            if not (min_separation <= separation <= max_separation):
                continue
            if (
                side_offsets[low_mask].std() > layer_std_max
                or side_offsets[~low_mask].std() > layer_std_max
            ):
                continue

            asymmetric = min(fraction, 1.0 - fraction) < min_layer_fraction
            if asymmetric and not allow_asymmetric_layers:
                continue
            minor_mask = low_mask if fraction <= 0.5 else ~low_mask
            minor_rows = side_rows[minor_mask]
            if asymmetric and (
                minor_rows.size < min_minor_layer_points
                or np.unique(all_pose[minor_rows]).size < min_minor_layer_keyframes
            ):
                continue

            overlap_ratio, overlap_cell_count, overlap_area_m2 = (
                _in_plane_overlap_metrics(
                side_points, low_mask, tangent_basis, overlap_bin_size
                )
            )
            if overlap_ratio < min_in_plane_overlap:
                continue

            poses_low, counts_low = np.unique(
                all_pose[side_rows[low_mask]], return_counts=True
            )
            poses_high, counts_high = np.unique(
                all_pose[side_rows[~low_mask]], return_counts=True
            )
            if (
                poses_low.size < min_layer_keyframes
                or poses_high.size < min_layer_keyframes
            ):
                continue
            shared = np.intersect1d(poses_low, poses_high)
            if shared.size > 0.5 * min(poses_low.size, poses_high.size):
                continue
            ranges_low = _contiguous_keyframe_ranges(poses_low, max_kf_gap)
            ranges_high = _contiguous_keyframe_ranges(poses_high, max_kf_gap)

            def range_support(
                pose_arr: np.ndarray, count_arr: np.ndarray, keyframe_range: list[int]
            ) -> tuple[int, int]:
                inside = (
                    (pose_arr >= keyframe_range[0])
                    & (pose_arr <= keyframe_range[1])
                )
                return int(count_arr[inside].sum()), int(inside.sum())

            best = None
            for range_a in ranges_low:
                for range_b in ranges_high:
                    if range_a[1] >= range_b[0] and range_b[1] >= range_a[0]:
                        continue
                    gap = abs(
                        (range_a[0] + range_a[1]) // 2
                        - (range_b[0] + range_b[1]) // 2
                    )
                    support_a, frame_count_a = range_support(
                        poses_low, counts_low, range_a
                    )
                    support_b, frame_count_b = range_support(
                        poses_high, counts_high, range_b
                    )
                    purity_a = support_a / max(int(counts_low.sum()), 1)
                    purity_b = support_b / max(int(counts_high.sum()), 1)
                    purity = min(purity_a, purity_b)
                    if purity < min_traversal_purity:
                        continue
                    if min(frame_count_a, frame_count_b) < min_layer_keyframes:
                        continue
                    score = (purity, min(support_a, support_b), gap)
                    if best is None or score > best[0]:
                        best = (
                            score,
                            gap,
                            range_a,
                            range_b,
                            purity_a,
                            purity_b,
                        )
            if best is None:
                continue
            _, gap, range_a, range_b, purity_a, purity_b = best
            if pose_count > max_kf_gap and gap <= max_kf_gap:
                continue

            def dominant(
                pose_arr: np.ndarray, count_arr: np.ndarray, keyframe_range: list[int]
            ) -> int:
                inside = (
                    (pose_arr >= keyframe_range[0])
                    & (pose_arr <= keyframe_range[1])
                )
                return int(pose_arr[inside][np.argmax(count_arr[inside])])

            voxel_key = sorted_keys[begin].astype(int).tolist()
            raw_regions.append(
                {
                    "candidate_type": "duplicated_surface_voxel",
                    "voxel_key": voxel_key,
                    "view_side": int(view_side),
                    "center_xyz": [round(float(value), 2) for value in side_points.mean(axis=0)],
                    "separation_m": round(float(separation), 3),
                    "normal": [round(float(value), 4) for value in normal],
                    "point_count": int(side_rows.size),
                    "asymmetric": bool(asymmetric),
                    "layer_support_fraction": round(min(fraction, 1.0 - fraction), 3),
                    "layer_overlap_ratio": round(overlap_ratio, 3),
                    "overlap_cell_count": overlap_cell_count,
                    "overlap_area_m2": round(overlap_area_m2, 3),
                    "traversal_purity": [round(purity_a, 3), round(purity_b, 3)],
                    "voxel_count": 1,
                    "layer_a_keyframes": range_a,
                    "layer_b_keyframes": range_b,
                    "suggested_pair": [
                        dominant(poses_low, counts_low, range_a),
                        dominant(poses_high, counts_high, range_b),
                    ],
                }
            )

    merged = _merge_connected_ghost_voxels(
        raw_regions,
        max_kf_gap=max_kf_gap,
        min_region_voxels=min_region_voxels,
        normal_cos_min=region_normal_cos_min,
        separation_tolerance=region_separation_tolerance,
        max_spatial_voxel_gap=max_spatial_voxel_gap,
        coplanar_tolerance=region_coplanar_tolerance,
    )
    merged = merged[:max_regions]
    # Only the worst ten are narrated however many are returned: callers that
    # want the full survey (the repair check) pass a max_regions in the
    # thousands, and printing all of them would bury the run log.
    for region in merged[:10]:
        _log(
            log_fn,
            f"[BALM] Residual duplicated surface ~"
            f"{region['separation_m'] * 100:.0f} cm near "
            f"({region['center_xyz'][0]:.0f}, {region['center_xyz'][1]:.0f}): "
            f"kf {region['layer_a_keyframes'][0]}-{region['layer_a_keyframes'][1]} vs "
            f"kf {region['layer_b_keyframes'][0]}-{region['layer_b_keyframes'][1]}; "
            f"overlap={region['overlap_area_m2']:.2f} m^2, diagnostic only.",
        )
    return merged
# ===== END CHANGE: balm plane bundle adjustment =====
