from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .concurrency import nn_query_workers

# Threads per cKDTree.query call. The search is exact, so the worker count
# cannot change which neighbour is returned; it only splits the query points.
# One is fastest because a gated ICP issues a query per iteration -- thirty for
# the near tier, a hundred for the wide one -- and each call sets up and tears
# down its own thread team. Worth 1.26-1.39x. See concurrency.py.
#
# What is NOT restored is hoisting the per-pair downsample/normals/KD-tree out
# of the hypothesis loop. That one changes what each registration is handed,
# and on the 2,579-keyframe in-house session it moved the result from
# 5-9-13-18 constraints to 5-9-12-16-20-25-30-34 at the same overlap floor.
# escalator00 and the simulation stayed byte-identical; at two rounds each they
# cannot see it.
_NN_WORKERS = nn_query_workers()

from .pcd_io import load_xyz_points
from .trajectory_io import TrajectoryData

# Per-keyframe point cache, module-level so it survives per-round workspace
# rebuilds. Local-frame geometry is pose-independent: the points never change,
# so each keyframe is read once regardless of how the trajectory evolves.
_FRAME_POINTS_CACHE: Dict[Tuple[str, int], np.ndarray] = {}
# Observation-oriented normals are pose-independent as well, but unlike the
# historical TRO implementation this cache is bounded.  A long sequence can
# contain tens of millions of raw points; retaining every float32 normal made
# an otherwise completed repair keep several extra gigabytes resident.
_FRAME_NORMALS_CACHE: OrderedDict[Tuple[str, int, int], np.ndarray] = OrderedDict()
# Production runs one Initial Loop Search, not the historical ten-round
# ghost loop.  Retaining 150 M float64 points (~3.6 GB) after a repair made an
# otherwise idle GUI look like it was leaking and left too little headroom for
# the 12-13 GiB BALM solve.  Thirty million points still cache a useful GICP
# working set (~0.72 GB) and can be overridden for benchmarks.
FRAME_CACHE_MAX_POINTS = int(os.environ.get(
    "GHOSTLOOP_FRAME_CACHE_MAX_POINTS", "30000000"
))
TARGET_CACHE_MAX_ENTRIES = max(1, int(os.environ.get(
    "GHOSTLOOP_TARGET_CACHE_MAX_ENTRIES", "2"
)))
NORMAL_CACHE_MAX_POINTS = max(0, int(os.environ.get(
    "GHOSTLOOP_NORMAL_CACHE_MAX_POINTS", "10000000"
)))
GATE_PREP_CACHE_MAX_ENTRIES = max(1, int(os.environ.get(
    "GHOSTLOOP_GATE_PREP_CACHE_MAX_ENTRIES", "1"
)))
_frame_cache_points_total = 0
_normal_cache_points_total = 0
# A pair's initial guesses register concurrently, and they read overlapping
# keyframes. Dict get/set are individually atomic under the GIL, but the budget
# is a read-modify-write and would drift; two threads racing to load the same
# frame is merely wasted work, so only the accounting is guarded.
_frame_cache_lock = threading.Lock()

# Legacy Open3D is faster for a single scan or a coarse outdoor reduction; its
# GIL-holding voxel hash becomes the UI bottleneck only on fine, merged maps.
TENSOR_VOXEL_MIN_INPUT_POINTS = 500_000
TENSOR_VOXEL_MAX_SIZE_M = 0.25

TARGET_CLOUD_MODE_TEMPORAL_WINDOW = "temporal_window"
TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP = "rs_spatial_submap"
TARGET_CLOUD_MODE_CHOICES = (
    TARGET_CLOUD_MODE_TEMPORAL_WINDOW,
    TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP,
)

OFFICE_DEFAULT_TARGET_NEIGHBORS = 40
OFFICE_DEFAULT_TARGET_MIN_TIME_GAP_SEC = 30.0
OFFICE_DEFAULT_TARGET_MAP_VOXEL_SIZE = 0.2
OFFICE_DEFAULT_TARGET_WINDOW = OFFICE_DEFAULT_TARGET_NEIGHBORS
OFFICE_DEFAULT_TARGET_CLOUD_MODE = TARGET_CLOUD_MODE_TEMPORAL_WINDOW
OFFICE_DEFAULT_VOXEL_SIZE = 0.0
OFFICE_DEFAULT_MAX_CORRESPONDENCE_DISTANCE = 2.0
OFFICE_DEFAULT_MAX_ITERATIONS = 30
# 回环因子的噪声。这个数唯一有意义的解读是它与**里程计边**的比值 —— 因子图的代价
# 函数对整体缩放不变，实测把两边同时放大 8 倍 ATE 一动不动（335.64 vs 335.74）。
# 里程计边的信息矩阵是 1000，即每条边 sigma = 1/sqrt(1000) = 0.0316 m。
#
# 一条跨 N 帧的回环不是在和一条边比，是在和 N 条边串起来的链比，后者累积 sigma 是
# 0.0316*sqrt(N)。所以比值决定了"回环要跨多长才压得过里程计"：
#
#     N* = (sigma_loop / 0.0316)^2
#
# 旧值 0.1（sigma 0.316，比值 10）对应 N* = 100 帧。MCD ntu 的 26 条约束里有 25 条
# 跨度超过 100 帧 —— 等于每一条回环都硬压里程计，而实测回环的测量误差中位 1.25 m，
# 于是这些误差被逐条精确地写进轨迹。
#
# 1.6（sigma 1.265，比值 40）把 N* 提到 1600 帧，只剩真正跨越大漂移的长回环说了算。
# 八条序列实测（只重解 PGO）：ntu 335.6 -> 213.9，kth 516.0 -> 513.0，sim 7.2 -> 6.7，
# hall04 4.6 -> 4.5，esc / hall02 / cc05 / bd 持平 —— 没有一条变差。
#
# This historical rotation value remains for old-project display and exact
# legacy reproductions.  New CSV exports convert its radian standard deviation
# to degrees explicitly; production Initial Loop Search uses the separately calibrated
# 0.5-degree profile in repair_presets.  The old path wrote sqrt(0.1) into a
# column labelled degrees and accidentally applied 0.316 degrees instead of
# 0.316 radians.
OFFICE_DEFAULT_VARIANCE_T = (1.6, 1.6, 1.6)
OFFICE_DEFAULT_VARIANCE_R_RAD2 = (0.1, 0.1, 0.1)


@dataclass(frozen=True)
class RegistrationConfig:
    target_cloud_mode: str = OFFICE_DEFAULT_TARGET_CLOUD_MODE
    target_neighbors: int = OFFICE_DEFAULT_TARGET_NEIGHBORS
    min_time_gap_sec: float = OFFICE_DEFAULT_TARGET_MIN_TIME_GAP_SEC
    target_map_voxel_size: float = OFFICE_DEFAULT_TARGET_MAP_VOXEL_SIZE
    voxel_size: float = OFFICE_DEFAULT_VOXEL_SIZE
    max_correspondence_distance: float = OFFICE_DEFAULT_MAX_CORRESPONDENCE_DISTANCE
    max_iterations: int = OFFICE_DEFAULT_MAX_ITERATIONS
    # When > 0, register a local submap of +-source_window keyframes (stitched
    # with relative odometry, which is locally accurate) instead of the single
    # source scan. Restores overlap for reverse-direction revisits, where a
    # single scan sees the opposite faces of everything.
    source_window: int = 0
    # Experimental M3 refinement.  It is intentionally opt-in: ordinary GICP
    # first supplies the basin, then observation-oriented signed normals stop
    # opposite faces of thin structures from becoming correspondences.
    normal_gate: bool = False
    normal_gate_max_angle_deg: float = 60.0
    normal_radius: float = 0.25


@dataclass(frozen=True)
class RegistrationPreview:
    source_id: int
    target_id: int
    target_cloud_mode: str
    target_neighbors: int
    min_time_gap_sec: float
    target_map_voxel_size: float
    target_frame_indices: tuple[int, ...]
    time_gap_filter_enabled: bool
    time_gap_filter_applied: bool
    target_points_world: np.ndarray
    source_points_local: np.ndarray
    source_points_world_initial: np.ndarray
    source_points_world_adjusted: np.ndarray
    transform_world_source_initial: np.ndarray
    transform_world_source_adjusted: np.ndarray
    transform_world_target: np.ndarray

    @property
    def target_frame_count(self) -> int:
        return len(self.target_frame_indices)

    @property
    def target_point_count(self) -> int:
        return int(self.target_points_world.shape[0])

    @property
    def target_frame_range(self) -> tuple[int, int] | None:
        if not self.target_frame_indices:
            return None
        return int(self.target_frame_indices[0]), int(self.target_frame_indices[-1])

    @property
    def target_window_clipped(self) -> bool:
        if self.target_cloud_mode != TARGET_CLOUD_MODE_TEMPORAL_WINDOW:
            return False
        expected_count = max(2 * int(self.target_neighbors) + 1, 1)
        return self.target_frame_count < expected_count


@dataclass(frozen=True)
class RegistrationResult:
    preview: RegistrationPreview
    transform_world_source_final: np.ndarray
    transform_target_source_final: np.ndarray
    source_points_world_final: np.ndarray
    fitness: float
    inlier_rmse: float
    oriented_diagnostics: "OrientedIcpDiagnostics | None" = None


@dataclass(frozen=True)
class OrientedIcpDiagnostics:
    """Separate geometric support from M3's signed-normal support.

    ``spatial_*`` uses every nearest neighbour inside the distance threshold,
    matching the ordinary overlap convention. ``oriented_*`` additionally
    requires the two normals to point in consistent observed directions.
    Keeping both prevents a low signed-normal support rate from being hidden
    inside a seemingly healthy geometric fitness value.
    """

    spatial_overlap: float
    spatial_rmse: float
    spatial_correspondence_count: int
    oriented_overlap: float
    oriented_rmse: float
    oriented_correspondence_count: int
    iterations: int
    valid: bool
    termination_reason: str


def build_delta_transform(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "xyz",
        [roll_deg, pitch_deg, yaw_deg],
        degrees=True,
    ).as_matrix()
    transform[:3, 3] = np.asarray([x, y, z], dtype=np.float64)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    rotated = points @ transform[:3, :3].T
    return rotated + transform[:3, 3]


def numpy_to_open3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if points.size:
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    return cloud


def voxel_downsample_indexed(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices of one representative point per occupied voxel (keeps sidecar
    arrays such as per-point observer positions aligned)."""
    if points.shape[0] == 0 or voxel_size <= 0:
        return np.arange(points.shape[0])
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def oriented_normals(
    points: np.ndarray,
    observers: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Estimate normals and orient each one toward its observing sensor."""
    if points.shape != observers.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and observers must both have shape (N, 3)")
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    cloud = numpy_to_open3d(points)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(max(float(radius), 1e-6))
    )
    normals = np.asarray(cloud.normals, dtype=np.float64).copy()
    flip = np.einsum("ij,ij->i", normals, observers - points) < 0.0
    normals[flip] = -normals[flip]
    return normals


def _solve_point_to_plane(
    source_world: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
) -> np.ndarray:
    """Return one damped, linearized point-to-plane SE(3) increment."""
    residuals = np.einsum("ij,ij->i", source_world - target, target_normals)
    jacobian = np.hstack(
        [np.cross(source_world, target_normals), target_normals]
    )
    hessian = jacobian.T @ jacobian + 1e-6 * np.eye(6)
    gradient = jacobian.T @ residuals
    update = -np.linalg.solve(hessian, gradient)
    if not np.all(np.isfinite(update)):
        raise FloatingPointError("M3 point-to-plane update is not finite")

    increment = np.eye(4, dtype=np.float64)
    angle = float(np.linalg.norm(update[:3]))
    if angle > 1e-12:
        axis = update[:3] / angle
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        increment[:3, :3] = (
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )
    increment[:3, 3] = update[3:]
    return increment


@dataclass(frozen=True)
class GatedIcpInputs:
    """Pose-independent inputs shared by M3 refinements of one pair."""

    source: np.ndarray
    source_normals: np.ndarray
    target: np.ndarray
    target_normals: np.ndarray
    tree: object


def prepare_gated_icp(
    source_local: np.ndarray,
    source_observers_local: np.ndarray,
    target_world: np.ndarray,
    target_observers_world: np.ndarray,
    *,
    voxel_size: float,
    normal_radius: float,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
) -> GatedIcpInputs:
    """Downsample geometry and aligned normal sidecars, then index target."""
    from scipy.spatial import cKDTree

    source_keep = voxel_downsample_indexed(source_local, voxel_size)
    target_keep = voxel_downsample_indexed(target_world, voxel_size)
    source = np.asarray(source_local[source_keep], dtype=np.float64)
    target = np.asarray(target_world[target_keep], dtype=np.float64)
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise RuntimeError("Empty cloud in observation-oriented ICP")

    if source_normals is None:
        source_normal_values = oriented_normals(
            source,
            source_observers_local[source_keep],
            normal_radius,
        )
    else:
        source_normal_values = np.asarray(
            source_normals, dtype=np.float64
        )[source_keep]
    if target_normals is None:
        target_normal_values = oriented_normals(
            target,
            target_observers_world[target_keep],
            normal_radius,
        )
    else:
        target_normal_values = np.asarray(
            target_normals, dtype=np.float64
        )[target_keep]

    return GatedIcpInputs(
        source=source,
        source_normals=source_normal_values,
        target=target,
        target_normals=target_normal_values,
        tree=cKDTree(target),
    )


def registration_normal_gated_icp(
    source_local: np.ndarray,
    source_observers_local: np.ndarray,
    target_world: np.ndarray,
    target_observers_world: np.ndarray,
    initial_transform_world_source: np.ndarray,
    *,
    voxel_size: float,
    max_correspondence_distance: float,
    max_iterations: int,
    normal_radius: float,
    max_angle_deg: float,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    prepared: GatedIcpInputs | None = None,
) -> tuple[np.ndarray, OrientedIcpDiagnostics]:
    """Refine a converged GICP measurement using signed normal agreement.

    M3 is deliberately a local refinement, not a global initializer.  The
    caller is expected to pass the final ordinary-GICP transform as ``initial``.
    Oppositely observed faces then cannot attract one another even when their
    unsigned surface normals are parallel.
    """
    if prepared is None:
        prepared = prepare_gated_icp(
            source_local,
            source_observers_local,
            target_world,
            target_observers_world,
            voxel_size=voxel_size,
            normal_radius=normal_radius,
            source_normals=source_normals,
            target_normals=target_normals,
        )
    source = prepared.source
    source_normal_values = prepared.source_normals
    target = prepared.target
    target_normal_values = prepared.target_normals
    cosine_gate = math.cos(math.radians(float(max_angle_deg)))

    transform = np.asarray(
        initial_transform_world_source, dtype=np.float64
    ).copy()
    iterations = 0
    termination_reason = "max_iterations"
    for iteration in range(max(0, int(max_iterations))):
        source_world = transform_points(source, transform)
        source_normals_world = source_normal_values @ transform[:3, :3].T
        distances, neighbors = prepared.tree.query(
            source_world,
            distance_upper_bound=float(max_correspondence_distance),
            workers=_NN_WORKERS,
        )
        matched = np.isfinite(distances)
        if not matched.any():
            termination_reason = "no_spatial_correspondences"
            break
        source_rows = np.nonzero(matched)[0]
        target_rows = neighbors[matched]
        normal_cosines = np.einsum(
            "ij,ij->i",
            source_normals_world[source_rows],
            target_normal_values[target_rows],
        )
        consistent = np.isfinite(normal_cosines) & (normal_cosines >= cosine_gate)
        if int(np.count_nonzero(consistent)) < 10:
            termination_reason = "insufficient_oriented_correspondences"
            break
        point_rows = source_rows[consistent]
        target_consistent = target_rows[consistent]
        try:
            increment = _solve_point_to_plane(
                source_world[point_rows],
                target[target_consistent],
                target_normal_values[target_consistent],
            )
        except (np.linalg.LinAlgError, FloatingPointError):
            termination_reason = "linear_solve_failed"
            break
        transform = increment @ transform
        iterations = iteration + 1
        if not np.all(np.isfinite(transform)):
            termination_reason = "non_finite_transform"
            transform = np.asarray(
                initial_transform_world_source, dtype=np.float64
            ).copy()
            break
        step = float(
            np.linalg.norm(increment[:3, 3])
            + np.linalg.norm(increment[:3, :3] - np.eye(3))
        )
        if step < 1e-7:
            termination_reason = "converged"
            break

    source_world = transform_points(source, transform)
    source_normals_world = source_normal_values @ transform[:3, :3].T
    distances, neighbors = prepared.tree.query(
        source_world,
        distance_upper_bound=float(max_correspondence_distance),
        workers=_NN_WORKERS,
    )
    matched = np.isfinite(distances)
    spatial_count = int(np.count_nonzero(matched))
    oriented_mask = np.zeros(source.shape[0], dtype=bool)
    if spatial_count:
        source_rows = np.nonzero(matched)[0]
        target_rows = neighbors[matched]
        normal_cosines = np.einsum(
            "ij,ij->i",
            source_normals_world[source_rows],
            target_normal_values[target_rows],
        )
        consistent = np.isfinite(normal_cosines) & (normal_cosines >= cosine_gate)
        oriented_mask[source_rows[consistent]] = True
    oriented_count = int(np.count_nonzero(oriented_mask))
    denominator = max(int(source.shape[0]), 1)
    spatial_rmse = (
        float(np.sqrt(np.mean(np.square(distances[matched]))))
        if spatial_count
        else float("inf")
    )
    oriented_rmse = (
        float(np.sqrt(np.mean(np.square(distances[oriented_mask]))))
        if oriented_count
        else float("inf")
    )
    valid = bool(
        oriented_count >= 10
        and np.all(np.isfinite(transform))
        and math.isfinite(oriented_rmse)
    )
    diagnostics = OrientedIcpDiagnostics(
        spatial_overlap=float(spatial_count / denominator),
        spatial_rmse=spatial_rmse,
        spatial_correspondence_count=spatial_count,
        oriented_overlap=float(oriented_count / denominator),
        oriented_rmse=oriented_rmse,
        oriented_correspondence_count=oriented_count,
        iterations=iterations,
        valid=valid,
        termination_reason=termination_reason,
    )
    return transform, diagnostics


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0 or voxel_size <= 0.0:
        return points.copy()
    if (points.shape[0] < TENSOR_VOXEL_MIN_INPUT_POINTS
            or voxel_size > TENSOR_VOXEL_MAX_SIZE_M):
        cloud = numpy_to_open3d(points)
        cloud = cloud.voxel_down_sample(voxel_size)
        return np.asarray(cloud.points, dtype=np.float64)
    # The legacy geometry implementation holds Python's GIL for the complete
    # voxel reduction.  A 1.5 M-point indoor target therefore froze Qt's event
    # loop for 0.75 s even though preview construction already ran in a
    # QThread.  Open3D's tensor CPU kernel is materially faster.  Shift by the
    # exact origin used by legacy ``voxel_down_sample`` so voxel membership and
    # registration semantics stay unchanged; only parallel summation order can
    # differ (31 micrometres maximum on the 744k-point GUI benchmark).  Keep
    # the legacy path for smaller/coarser clouds where tensor setup costs more.
    origin = np.min(points, axis=0) - 0.5 * float(voxel_size)
    shifted = np.asarray(points - origin, dtype=np.float64)
    cloud = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(shifted, dtype=o3d.core.Dtype.Float64)
    )
    downsampled = cloud.voxel_down_sample(float(voxel_size))
    return downsampled.point.positions.numpy() + origin


class RegistrationWorkspace:
    def __init__(self, keyframe_dir: Path, trajectory: TrajectoryData) -> None:
        self._keyframe_dir = keyframe_dir
        self._trajectory = trajectory
        self._local_cache: Dict[int, np.ndarray] = {}
        self._target_cache: OrderedDict[
            Tuple[str, int, int, int, float, float],
            tuple[tuple[int, ...], bool, np.ndarray],
        ] = OrderedDict()
        self._gate_prep_cache: OrderedDict[tuple, GatedIcpInputs] = OrderedDict()
        self._gate_prep_lock = threading.Lock()

    @property
    def trajectory(self) -> TrajectoryData:
        return self._trajectory

    def load_local_points(self, index: int) -> np.ndarray:
        key = (str(self._keyframe_dir), int(index))
        cached = _FRAME_POINTS_CACHE.get(key)
        if cached is None:
            global _frame_cache_points_total
            path = self._keyframe_dir / f"{index}.pcd"
            cached = load_xyz_points(path)
            with _frame_cache_lock:
                if _frame_cache_points_total + cached.shape[0] <= FRAME_CACHE_MAX_POINTS:
                    _FRAME_POINTS_CACHE[key] = cached
                    _frame_cache_points_total += cached.shape[0]
        return cached

    def load_local_points_uncached(self, index: int) -> np.ndarray:
        """Read one frame without populating the multi-gigabyte GICP cache.

        Sequential whole-session passes such as descriptor construction touch
        every keyframe only once. Caching those raw clouds evicts useful GICP
        working sets and can add several gigabytes to the prepared diagnosis
        stack, so those passes should use this explicit entry point.
        """
        path = self._keyframe_dir / f"{index}.pcd"
        return load_xyz_points(path)

    def local_oriented_normals(self, index: int, radius: float) -> np.ndarray:
        """Viewpoint-oriented local normals with a bounded shared LRU cache."""
        key = (
            str(self._keyframe_dir),
            int(index),
            int(round(float(radius) * 1000.0)),
        )
        with _frame_cache_lock:
            cached = _FRAME_NORMALS_CACHE.get(key)
            if cached is not None:
                _FRAME_NORMALS_CACHE.move_to_end(key)
                return cached

        points = self.load_local_points(index)
        normals = oriented_normals(
            points,
            np.zeros_like(points),
            radius,
        ).astype(np.float32)

        global _normal_cache_points_total
        with _frame_cache_lock:
            existing = _FRAME_NORMALS_CACHE.get(key)
            if existing is not None:
                _FRAME_NORMALS_CACHE.move_to_end(key)
                return existing
            point_count = int(normals.shape[0])
            while (
                _FRAME_NORMALS_CACHE
                and _normal_cache_points_total + point_count
                > NORMAL_CACHE_MAX_POINTS
            ):
                _, evicted = _FRAME_NORMALS_CACHE.popitem(last=False)
                _normal_cache_points_total -= int(evicted.shape[0])
            if point_count <= NORMAL_CACHE_MAX_POINTS:
                _FRAME_NORMALS_CACHE[key] = normals
                _normal_cache_points_total += point_count
        return normals

    def clear_caches(self, *, include_shared_frames: bool = False) -> None:
        """Release pose-dependent targets and optionally this session's frames.

        Automatic Repair calls this at its terminal boundary.  Manual use can
        simply reload the data on demand, while BALM and the next session no
        longer compete with stale GICP arrays for memory.
        """
        self._local_cache.clear()
        self._target_cache.clear()
        with self._gate_prep_lock:
            self._gate_prep_cache.clear()
        if not include_shared_frames:
            return
        prefix = str(self._keyframe_dir)
        global _frame_cache_points_total, _normal_cache_points_total
        with _frame_cache_lock:
            doomed = [key for key in _FRAME_POINTS_CACHE if key[0] == prefix]
            released = sum(
                int(_FRAME_POINTS_CACHE[key].shape[0]) for key in doomed
            )
            for key in doomed:
                _FRAME_POINTS_CACHE.pop(key, None)
            _frame_cache_points_total = max(
                0, int(_frame_cache_points_total) - released
            )
            normal_keys = [
                key for key in _FRAME_NORMALS_CACHE if key[0] == prefix
            ]
            normal_released = sum(
                int(_FRAME_NORMALS_CACHE[key].shape[0])
                for key in normal_keys
            )
            for key in normal_keys:
                _FRAME_NORMALS_CACHE.pop(key, None)
            _normal_cache_points_total = max(
                0, int(_normal_cache_points_total) - normal_released
            )

    def _target_cache_key(
        self,
        *,
        target_cloud_mode: str,
        source_id: int,
        target_id: int,
        target_neighbors: int,
        min_time_gap_sec: float,
        target_map_voxel_size: float,
    ) -> tuple[str, int, int, int, float, float]:
        return (
            str(target_cloud_mode),
            int(source_id),
            int(target_id),
            int(target_neighbors),
            round(float(min_time_gap_sec), 6),
            round(float(target_map_voxel_size), 6),
        )

    def select_temporal_window_frame_indices(
        self,
        *,
        target_id: int,
        target_window_radius: int,
    ) -> tuple[tuple[int, ...], bool]:
        window_radius = max(int(target_window_radius), 0)
        start_index = max(int(target_id) - window_radius, 0)
        end_index = min(int(target_id) + window_radius, self._trajectory.size - 1)
        selected = tuple(range(start_index, end_index + 1))
        if not selected:
            raise RuntimeError("Temporal target window does not contain any keyframes.")
        return selected, False

    def select_rs_spatial_frame_indices(
        self,
        *,
        source_id: int,
        target_id: int,
        target_neighbors: int,
        min_time_gap_sec: float,
    ) -> tuple[tuple[int, ...], bool]:
        target_xy = self._trajectory.positions_xyz[target_id, :2]
        source_timestamp = float(self._trajectory.timestamps[source_id])
        candidates: list[tuple[float, int]] = []
        filtered_due_to_time_gap = 0

        for index in range(self._trajectory.size):
            time_difference = abs(float(self._trajectory.timestamps[index]) - source_timestamp)
            if min_time_gap_sec > 0.0 and time_difference < min_time_gap_sec:
                filtered_due_to_time_gap += 1
                continue

            pose_xy = self._trajectory.positions_xyz[index, :2]
            distance = float(np.linalg.norm(pose_xy - target_xy))
            candidates.append((distance, index))

        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = tuple(index for _, index in candidates[: max(int(target_neighbors), 1)])
        if not selected:
            raise RuntimeError(
                "No target frames remain after applying the RS loop-closure time-gap filter."
            )
        return selected, filtered_due_to_time_gap > 0

    def select_target_frame_indices(
        self,
        *,
        target_cloud_mode: str,
        source_id: int,
        target_id: int,
        target_neighbors: int,
        min_time_gap_sec: float,
    ) -> tuple[tuple[int, ...], bool]:
        if target_cloud_mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW:
            return self.select_temporal_window_frame_indices(
                target_id=target_id,
                target_window_radius=target_neighbors,
            )
        if target_cloud_mode == TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP:
            return self.select_rs_spatial_frame_indices(
                source_id=source_id,
                target_id=target_id,
                target_neighbors=target_neighbors,
                min_time_gap_sec=min_time_gap_sec,
            )
        raise RuntimeError(f"Unsupported target cloud mode: {target_cloud_mode}")

    def build_target_submap(
        self,
        *,
        target_cloud_mode: str,
        source_id: int,
        target_id: int,
        target_neighbors: int,
        min_time_gap_sec: float,
        target_map_voxel_size: float,
    ) -> tuple[tuple[int, ...], bool, np.ndarray]:
        key = self._target_cache_key(
            target_cloud_mode=target_cloud_mode,
            source_id=source_id,
            target_id=target_id,
            target_neighbors=target_neighbors,
            min_time_gap_sec=min_time_gap_sec,
            target_map_voxel_size=target_map_voxel_size,
        )
        if key in self._target_cache:
            self._target_cache.move_to_end(key)
            return self._target_cache[key]

        selected_indices, time_gap_filter_applied = self.select_target_frame_indices(
            target_cloud_mode=target_cloud_mode,
            source_id=source_id,
            target_id=target_id,
            target_neighbors=target_neighbors,
            min_time_gap_sec=min_time_gap_sec,
        )

        merged_chunks = []
        for index in selected_indices:
            local_points = self.load_local_points(index)
            world_points = transform_points(
                local_points,
                self._trajectory.transforms_world_sensor[index],
            )
            merged_chunks.append(world_points)

        merged = (
            np.concatenate(merged_chunks, axis=0)
            if merged_chunks
            else np.empty((0, 3), dtype=np.float64)
        )
        merged = voxel_downsample(merged, target_map_voxel_size)
        result = (selected_indices, time_gap_filter_applied, merged)
        self._target_cache[key] = result
        self._target_cache.move_to_end(key)
        while len(self._target_cache) > TARGET_CACHE_MAX_ENTRIES:
            self._target_cache.popitem(last=False)
        return result

    def build_preview(
        self,
        *,
        source_id: int,
        target_id: int,
        delta_transform_local: np.ndarray,
        target_cloud_mode: str,
        target_neighbors: int,
        min_time_gap_sec: float,
        target_map_voxel_size: float,
    ) -> RegistrationPreview:
        source_local = self.load_local_points(source_id)
        target_frame_indices, time_gap_filter_applied, target_points_world = self.build_target_submap(
            target_cloud_mode=target_cloud_mode,
            source_id=source_id,
            target_id=target_id,
            target_neighbors=target_neighbors,
            min_time_gap_sec=min_time_gap_sec,
            target_map_voxel_size=target_map_voxel_size,
        )
        transform_world_source_initial = self._trajectory.transforms_world_sensor[source_id]
        # Apply the manual delta in the current source local frame, then move the
        # adjusted cloud into map coordinates.
        transform_world_source_adjusted = transform_world_source_initial @ delta_transform_local
        transform_world_target = self._trajectory.transforms_world_sensor[target_id]

        return RegistrationPreview(
            source_id=source_id,
            target_id=target_id,
            target_cloud_mode=str(target_cloud_mode),
            target_neighbors=int(target_neighbors),
            min_time_gap_sec=float(min_time_gap_sec),
            target_map_voxel_size=float(target_map_voxel_size),
            target_frame_indices=target_frame_indices,
            time_gap_filter_enabled=(
                target_cloud_mode == TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP
                and min_time_gap_sec > 0.0
            ),
            time_gap_filter_applied=time_gap_filter_applied,
            target_points_world=target_points_world,
            source_points_local=source_local,
            source_points_world_initial=transform_points(
                source_local,
                transform_world_source_initial,
            ),
            source_points_world_adjusted=transform_points(
                source_local,
                transform_world_source_adjusted,
            ),
            transform_world_source_initial=transform_world_source_initial,
            transform_world_source_adjusted=transform_world_source_adjusted,
            transform_world_target=transform_world_target,
        )

    def build_source_submap(self, source_id: int, window: int) -> np.ndarray:
        """Neighbor keyframes stitched into the source's local frame."""
        transforms = self.trajectory.transforms_world_sensor
        inv_source = np.linalg.inv(transforms[source_id])
        lo = max(0, source_id - window)
        hi = min(self.trajectory.size - 1, source_id + window)
        parts = []
        for index in range(lo, hi + 1):
            points = self.load_local_points(index)
            relative = inv_source @ transforms[index]
            parts.append(points @ relative[:3, :3].T + relative[:3, 3])
        return np.vstack(parts)

    def build_source_cloud_with_observers(
        self,
        source_id: int,
        window: int,
        *,
        normal_radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a source-local submap with per-point observers and normals."""
        transforms = self.trajectory.transforms_world_sensor
        inverse_source = np.linalg.inv(transforms[source_id])
        lower = max(0, int(source_id) - max(0, int(window)))
        upper = min(
            self.trajectory.size - 1,
            int(source_id) + max(0, int(window)),
        )
        point_parts: list[np.ndarray] = []
        observer_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        for index in range(lower, upper + 1):
            points = self.load_local_points(index)
            relative = inverse_source @ transforms[index]
            point_parts.append(transform_points(points, relative))
            observer_parts.append(
                np.repeat(relative[None, :3, 3], points.shape[0], axis=0)
            )
            normal_parts.append(
                self.local_oriented_normals(index, normal_radius)
                @ relative[:3, :3].T.astype(np.float32)
            )
        return (
            np.vstack(point_parts),
            np.vstack(observer_parts),
            np.vstack(normal_parts),
        )

    def build_target_cloud_with_observers(
        self,
        frame_indices: tuple[int, ...],
        *,
        normal_radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the raw world target with its observation-oriented normals."""
        transforms = self.trajectory.transforms_world_sensor
        point_parts: list[np.ndarray] = []
        observer_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        for index in frame_indices:
            points = self.load_local_points(index)
            transform = transforms[index]
            world = transform_points(points, transform)
            point_parts.append(world)
            observer_parts.append(
                np.repeat(transform[None, :3, 3], world.shape[0], axis=0)
            )
            normal_parts.append(
                self.local_oriented_normals(index, normal_radius)
                @ transform[:3, :3].T.astype(np.float32)
            )
        if not point_parts:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty.copy(), empty.copy()
        return (
            np.vstack(point_parts),
            np.vstack(observer_parts),
            np.vstack(normal_parts),
        )

    def _prepared_gate_inputs(
        self,
        *,
        source_id: int,
        source_window: int,
        target_frame_indices: tuple[int, ...],
        normal_radius: float,
        voxel_size: float,
    ) -> GatedIcpInputs:
        key = (
            int(source_id),
            int(source_window),
            target_frame_indices,
            round(float(normal_radius), 6),
            round(float(voxel_size), 6),
        )
        with self._gate_prep_lock:
            cached = self._gate_prep_cache.get(key)
            if cached is not None:
                self._gate_prep_cache.move_to_end(key)
                return cached
            source, source_observers, source_normals = (
                self.build_source_cloud_with_observers(
                    source_id,
                    max(0, source_window),
                    normal_radius=normal_radius,
                )
            )
            target, target_observers, target_normals = (
                self.build_target_cloud_with_observers(
                    target_frame_indices,
                    normal_radius=normal_radius,
                )
            )
            prepared = prepare_gated_icp(
                source,
                source_observers,
                target,
                target_observers,
                voxel_size=voxel_size,
                normal_radius=normal_radius,
                source_normals=source_normals,
                target_normals=target_normals,
            )
            self._gate_prep_cache[key] = prepared
            self._gate_prep_cache.move_to_end(key)
            while len(self._gate_prep_cache) > GATE_PREP_CACHE_MAX_ENTRIES:
                self._gate_prep_cache.popitem(last=False)
            return prepared

    def run_gicp(
        self,
        *,
        source_id: int,
        target_id: int,
        delta_transform_local: np.ndarray,
        config: RegistrationConfig,
    ) -> RegistrationResult:
        preview = self.build_preview(
            source_id=source_id,
            target_id=target_id,
            delta_transform_local=delta_transform_local,
            target_cloud_mode=config.target_cloud_mode,
            target_neighbors=config.target_neighbors,
            min_time_gap_sec=config.min_time_gap_sec,
            target_map_voxel_size=config.target_map_voxel_size,
        )

        if config.normal_gate:
            # A non-zero voxel is required for the raw multi-frame target.
            # Preserve the TRO fallback for legacy configurations that used
            # voxel_size=0 while making the mode explicitly opt-in.
            gated_voxel = float(config.voxel_size) if config.voxel_size > 0 else 0.05
            prepared = self._prepared_gate_inputs(
                source_id=source_id,
                source_window=max(0, config.source_window),
                target_frame_indices=preview.target_frame_indices,
                normal_radius=config.normal_radius,
                voxel_size=gated_voxel,
            )
            transform_world_source_final, diagnostics = (
                registration_normal_gated_icp(
                    prepared.source,
                    np.zeros_like(prepared.source),
                    prepared.target,
                    np.zeros_like(prepared.target),
                    preview.transform_world_source_adjusted,
                    voxel_size=gated_voxel,
                    max_correspondence_distance=(
                        config.max_correspondence_distance
                    ),
                    max_iterations=config.max_iterations,
                    normal_radius=config.normal_radius,
                    max_angle_deg=config.normal_gate_max_angle_deg,
                    prepared=prepared,
                )
            )
            transform_target_source_final = (
                np.linalg.inv(preview.transform_world_target)
                @ transform_world_source_final
            )
            return RegistrationResult(
                preview=preview,
                transform_world_source_final=transform_world_source_final,
                transform_target_source_final=transform_target_source_final,
                source_points_world_final=transform_points(
                    preview.source_points_local,
                    transform_world_source_final,
                ),
                fitness=diagnostics.spatial_overlap,
                inlier_rmse=diagnostics.spatial_rmse,
                oriented_diagnostics=diagnostics,
            )

        source_points_local = preview.source_points_local
        if config.source_window > 0:
            source_points_local = self.build_source_submap(
                source_id, config.source_window
            )
        source_down = voxel_downsample(source_points_local, config.voxel_size)
        target_down = voxel_downsample(preview.target_points_world, config.voxel_size)
        if source_down.size == 0 or target_down.size == 0:
            raise RuntimeError("Source or target point cloud is empty after downsampling.")

        source_cloud = numpy_to_open3d(source_down)
        target_cloud = numpy_to_open3d(target_down)

        result = o3d.pipelines.registration.registration_generalized_icp(
            source_cloud,
            target_cloud,
            config.max_correspondence_distance,
            preview.transform_world_source_adjusted,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=int(config.max_iterations),
            ),
        )

        transform_world_source_final = np.asarray(result.transformation, dtype=np.float64)
        transform_target_source_final = (
            np.linalg.inv(preview.transform_world_target) @ transform_world_source_final
        )

        return RegistrationResult(
            preview=preview,
            transform_world_source_final=transform_world_source_final,
            transform_target_source_final=transform_target_source_final,
            source_points_world_final=transform_points(
                preview.source_points_local,
                transform_world_source_final,
            ),
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
        )
