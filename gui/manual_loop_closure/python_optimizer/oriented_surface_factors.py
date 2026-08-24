"""Observation-oriented ICP and anisotropic pose-graph factor construction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .information_order import gtsam_information_to_g2o


@dataclass(frozen=True)
class OrientedFactorConfig:
    voxel_size_m: float = 0.20
    normal_radius_m: float = 0.60
    normal_max_nn: int = 40
    max_correspondence_m: float = 0.60
    normal_gate_deg: float = 45.0
    # Search several spatial neighbours before applying the oriented-normal
    # gate.  A nearest-neighbour-then-reject implementation is unsafe around
    # thin structure: the closest sample may lie on the opposite physical
    # face, while the correct same-side sample is only the second (or later)
    # spatial neighbour.  The first orientation-compatible neighbour retains
    # the usual nearest-neighbour objective within the admissible identity.
    normal_search_k: int = 32
    max_iterations: int = 30
    min_correspondences: int = 80
    min_overlap: float = 0.20
    huber_delta_m: float = 0.10
    residual_noise_floor_m: float = 0.015
    max_effective_correspondences: int = 300
    information_scale: float = 1.0
    # A point-cloud Hessian describes local geometric observability, but it
    # cannot make shared calibration/material/initialization bias vanish by
    # accumulating more correspondences.  These optional floors cap the final
    # factor confidence after Hessian scaling.  Defaults preserve legacy runs.
    translation_sigma_floor_m: float = 0.0
    rotation_sigma_floor_rad: float = 0.0
    degeneracy_ratio: float = 1e-4
    min_information_ratio: float = 1e-7
    convergence_translation_m: float = 1e-5
    convergence_rotation_rad: float = 1e-5


@dataclass(frozen=True)
class OrientedFactorResult:
    valid: bool
    reason: str
    transform_target_source: np.ndarray
    information_g2o: np.ndarray
    covariance_g2o: np.ndarray
    overlap: float
    spatial_overlap: float
    inlier_rmse_m: float
    residual_scale_m: float
    correspondence_count: int
    orientation_consistent_count: int
    observable_rank: int
    scaled_information_eigenvalues: np.ndarray
    scaled_condition_number: float
    iterations: int

    def report(self) -> dict:
        output = asdict(self)
        for key in (
            "transform_target_source", "information_g2o", "covariance_g2o",
            "scaled_information_eigenvalues",
        ):
            output[key] = np.asarray(output[key]).tolist()
        output["information_order"] = "g2o_translation_rotation"
        output["measurement_convention"] = "target_to_source"
        return output


@dataclass(frozen=True)
class PreparedOrientedCloud:
    points: np.ndarray
    normals: np.ndarray


def voxel_downsample_indices(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_size_m <= 0.0:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.concatenate([
        np.asarray([True]),
        np.any(np.diff(sorted_keys, axis=0) != 0, axis=1),
    ])
    return np.sort(order[first])


def normals_oriented_to_observer(
    points: np.ndarray,
    observer_xyz: np.ndarray,
    *,
    radius_m: float,
    max_nn: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals and orient every valid one toward its observing sensor."""
    points = np.asarray(points, dtype=np.float64)
    observer = np.asarray(observer_xyz, dtype=np.float64)
    if observer.shape == (3,):
        observer = np.broadcast_to(observer, points.shape)
    if observer.shape != points.shape:
        raise ValueError("observer_xyz must be one 3-vector or one per point")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius_m), max_nn=int(max_nn)
        )
    )
    normals = np.asarray(cloud.normals, dtype=np.float64).copy()
    norm = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(norm) & (norm > 0.5)
    normals[valid] /= norm[valid, None]
    flip = valid & (
        np.einsum("ij,ij->i", normals, observer - points) < 0.0
    )
    normals[flip] *= -1.0
    return normals, valid


def prepare_oriented_cloud(
    local_points: np.ndarray,
    config: OrientedFactorConfig = OrientedFactorConfig(),
) -> PreparedOrientedCloud:
    """Prepare one keyframe once for reuse by adjacent and loop factors."""
    points = np.asarray(local_points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[voxel_downsample_indices(points, config.voxel_size_m)]
    normals, valid = normals_oriented_to_observer(
        points, np.zeros(3), radius_m=config.normal_radius_m,
        max_nn=config.normal_max_nn,
    )
    return PreparedOrientedCloud(points[valid], normals[valid])


def _se3_increment(rotation_vector: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    transform[:3, 3] = translation
    return transform


def _huber_weights(residuals: np.ndarray, delta_m: float) -> np.ndarray:
    absolute = np.abs(residuals)
    weights = np.ones_like(absolute)
    outside = absolute > delta_m
    weights[outside] = delta_m / np.maximum(absolute[outside], 1e-12)
    return weights


def _mad_scale(residuals: np.ndarray, floor_m: float) -> float:
    if not len(residuals):
        return float(floor_m)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return max(float(floor_m), 1.4826 * mad)


def _correspondences(
    source: np.ndarray,
    source_normals: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    target_tree: cKDTree,
    transform_target_source: np.ndarray,
    config: OrientedFactorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    rotation = transform_target_source[:3, :3]
    source_target = source @ rotation.T + transform_target_source[:3, 3]
    source_normals_target = source_normals @ rotation.T
    search_k = max(1, min(int(config.normal_search_k), len(target)))
    distance, index = target_tree.query(
        source_target, k=search_k,
        distance_upper_bound=config.max_correspondence_m, workers=-1,
    )
    if search_k == 1:
        distance = distance[:, None]
        index = index[:, None]
    valid_candidate = np.isfinite(distance) & (index < len(target))
    spatial = valid_candidate.any(axis=1)
    spatial_count = int(spatial.sum())
    if spatial_count == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, empty, 0, 0.0, 0.0
    safe_index = np.minimum(index, len(target) - 1)
    cosine = np.einsum(
        "ij,ikj->ik", source_normals_target, target_normals[safe_index]
    )
    admissible = (
        valid_candidate
        & (cosine >= math.cos(math.radians(config.normal_gate_deg)))
    )
    oriented = admissible.any(axis=1)
    source_rows = np.nonzero(oriented)[0]
    # cKDTree returns candidates in ascending distance. argmax therefore gives
    # the nearest admissible surface sample, not an arbitrary normal match.
    chosen_column = np.argmax(admissible[source_rows], axis=1)
    target_rows = index[source_rows, chosen_column]
    spatial_overlap = spatial_count / max(len(source), 1)
    oriented_overlap = int(oriented.sum()) / max(len(source), 1)
    return (
        source_target[source_rows], target[target_rows], target_normals[target_rows],
        int(oriented.sum()), float(oriented_overlap), float(spatial_overlap),
    )


def _linear_system(
    transformed_source: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    config: OrientedFactorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residuals = np.einsum(
        "ij,ij->i", transformed_source - target, target_normals
    )
    # Left perturbation in target frame, GTSAM tangent order [rot, trans].
    jacobian = np.hstack([
        np.cross(transformed_source, target_normals), target_normals
    ])
    weights = _huber_weights(residuals, config.huber_delta_m)
    sqrt_weights = np.sqrt(weights)
    weighted_jacobian = jacobian * sqrt_weights[:, None]
    weighted_residual = residuals * sqrt_weights
    hessian = weighted_jacobian.T @ weighted_jacobian
    gradient = weighted_jacobian.T @ weighted_residual
    return hessian, gradient, residuals, weights


def _invalid_result(
    reason: str, transform: np.ndarray, *, overlap: float = 0.0,
    spatial_overlap: float = 0.0,
    count: int = 0, oriented_count: int = 0, iterations: int = 0,
) -> OrientedFactorResult:
    weak_information = np.eye(6, dtype=np.float64) * 1e-9
    return OrientedFactorResult(
        False, reason, transform.copy(), weak_information,
        np.eye(6, dtype=np.float64) * 1e9, overlap, spatial_overlap,
        float("inf"), float("inf"),
        count, oriented_count, 0, np.zeros(6), float("inf"), iterations,
    )


def estimate_oriented_surface_factor(
    source_local: np.ndarray,
    target_local: np.ndarray,
    initial_target_source: np.ndarray,
    config: OrientedFactorConfig = OrientedFactorConfig(),
    *,
    source_normals_local: np.ndarray | None = None,
    target_normals_local: np.ndarray | None = None,
) -> OrientedFactorResult:
    """Estimate target-to-source measurement and calibrated anisotropic weight."""
    if not np.isfinite(config.information_scale) or config.information_scale <= 0.0:
        raise ValueError("information_scale must be finite and positive")
    if (
        not np.isfinite(config.translation_sigma_floor_m)
        or config.translation_sigma_floor_m < 0.0
    ):
        raise ValueError("translation_sigma_floor_m must be finite and non-negative")
    if (
        not np.isfinite(config.rotation_sigma_floor_rad)
        or config.rotation_sigma_floor_rad < 0.0
    ):
        raise ValueError("rotation_sigma_floor_rad must be finite and non-negative")
    source_all = np.asarray(source_local, dtype=np.float64)
    target_all = np.asarray(target_local, dtype=np.float64)
    source_finite = np.isfinite(source_all).all(axis=1)
    target_finite = np.isfinite(target_all).all(axis=1)
    source_all = source_all[source_finite]
    target_all = target_all[target_finite]
    supplied_source_normals = None
    supplied_target_normals = None
    if source_normals_local is not None:
        supplied_source_normals = np.asarray(
            source_normals_local, dtype=np.float64
        )[source_finite]
    if target_normals_local is not None:
        supplied_target_normals = np.asarray(
            target_normals_local, dtype=np.float64
        )[target_finite]
    source_keep = voxel_downsample_indices(source_all, config.voxel_size_m)
    target_keep = voxel_downsample_indices(target_all, config.voxel_size_m)
    source = source_all[source_keep]
    target = target_all[target_keep]
    if len(source) < config.min_correspondences or len(target) < config.min_correspondences:
        return _invalid_result("too_few_downsampled_points", initial_target_source)
    if supplied_source_normals is None:
        source_normals, source_valid = normals_oriented_to_observer(
            source, np.zeros(3), radius_m=config.normal_radius_m,
            max_nn=config.normal_max_nn,
        )
    else:
        source_normals = supplied_source_normals[source_keep]
        source_norm = np.linalg.norm(source_normals, axis=1)
        source_valid = np.isfinite(source_norm) & (source_norm > 0.5)
        source_normals[source_valid] /= source_norm[source_valid, None]
    if supplied_target_normals is None:
        target_normals, target_valid = normals_oriented_to_observer(
            target, np.zeros(3), radius_m=config.normal_radius_m,
            max_nn=config.normal_max_nn,
        )
    else:
        target_normals = supplied_target_normals[target_keep]
        target_norm = np.linalg.norm(target_normals, axis=1)
        target_valid = np.isfinite(target_norm) & (target_norm > 0.5)
        target_normals[target_valid] /= target_norm[target_valid, None]
    source, source_normals = source[source_valid], source_normals[source_valid]
    target, target_normals = target[target_valid], target_normals[target_valid]
    if len(source) < config.min_correspondences or len(target) < config.min_correspondences:
        return _invalid_result("too_few_valid_normals", initial_target_source)
    tree = cKDTree(target)
    transform = np.asarray(initial_target_source, dtype=np.float64).copy()
    final_system = None
    overlap = 0.0
    spatial_overlap = 0.0
    oriented_count = 0
    iteration_count = 0
    for iteration in range(config.max_iterations):
        (transformed, matched, normals, oriented_count, overlap,
         spatial_overlap) = _correspondences(
            source, source_normals, target, target_normals, tree, transform, config
        )
        iteration_count = iteration + 1
        if oriented_count < config.min_correspondences:
            return _invalid_result(
                "too_few_oriented_correspondences", transform, overlap=overlap,
                spatial_overlap=spatial_overlap,
                count=len(source), oriented_count=oriented_count,
                iterations=iteration_count,
            )
        hessian, gradient, residuals, weights = _linear_system(
            transformed, matched, normals, config
        )
        final_system = (hessian, residuals, weights, transformed)
        scene_radius = max(
            1.0, float(np.median(np.linalg.norm(transformed - np.median(
                transformed, axis=0), axis=1)))
        )
        scale = np.diag([scene_radius, scene_radius, scene_radius, 1.0, 1.0, 1.0])
        scaled_hessian = np.linalg.inv(scale).T @ hessian @ np.linalg.inv(scale)
        damping = max(float(np.trace(scaled_hessian)) / 6.0, 1.0) * 1e-8
        try:
            delta_scaled = -np.linalg.solve(
                scaled_hessian + damping * np.eye(6),
                np.linalg.inv(scale).T @ gradient,
            )
        except np.linalg.LinAlgError:
            return _invalid_result(
                "singular_registration_system", transform, overlap=overlap,
                count=len(source), oriented_count=oriented_count,
                iterations=iteration_count,
            )
        delta = np.linalg.inv(scale) @ delta_scaled
        increment = _se3_increment(delta[:3], delta[3:])
        transform = increment @ transform
        if (
            np.linalg.norm(delta[:3]) <= config.convergence_rotation_rad
            and np.linalg.norm(delta[3:]) <= config.convergence_translation_m
        ):
            break

    (transformed, matched, normals, oriented_count, overlap,
     spatial_overlap) = _correspondences(
        source, source_normals, target, target_normals, tree, transform, config
    )
    if oriented_count < config.min_correspondences or overlap < config.min_overlap:
        return _invalid_result(
            "insufficient_final_support", transform, overlap=overlap,
            spatial_overlap=spatial_overlap,
            count=len(source), oriented_count=oriented_count,
            iterations=iteration_count,
        )
    hessian, _gradient, residuals, weights = _linear_system(
        transformed, matched, normals, config
    )
    residual_scale = _mad_scale(residuals, config.residual_noise_floor_m)
    effective_count = min(float(weights.sum()), float(config.max_effective_correspondences))
    normalized = hessian / max(float(weights.sum()), 1.0)
    information_gtsam = (
        normalized * effective_count / (residual_scale**2)
        * float(config.information_scale)
    )

    scene_radius = max(
        1.0, float(np.median(np.linalg.norm(transformed - np.median(
            transformed, axis=0), axis=1)))
    )
    scale = np.diag([scene_radius, scene_radius, scene_radius, 1.0, 1.0, 1.0])
    inverse_scale = np.linalg.inv(scale)
    scaled = inverse_scale.T @ information_gtsam @ inverse_scale
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scaled + scaled.T))
    maximum = max(float(eigenvalues[-1]), 1e-12)
    observable_rank = int(np.sum(eigenvalues >= config.degeneracy_ratio * maximum))
    floor = config.min_information_ratio * maximum
    regularized_eigenvalues = np.maximum(eigenvalues, floor)
    scaled_regularized = (
        eigenvectors @ np.diag(regularized_eigenvalues) @ eigenvectors.T
    )
    information_gtsam = scale.T @ scaled_regularized @ scale
    information_gtsam = 0.5 * (information_gtsam + information_gtsam.T)
    covariance_gtsam = np.linalg.inv(information_gtsam)
    covariance_gtsam += np.diag([
        float(config.rotation_sigma_floor_rad) ** 2,
        float(config.rotation_sigma_floor_rad) ** 2,
        float(config.rotation_sigma_floor_rad) ** 2,
        float(config.translation_sigma_floor_m) ** 2,
        float(config.translation_sigma_floor_m) ** 2,
        float(config.translation_sigma_floor_m) ** 2,
    ])
    covariance_gtsam = 0.5 * (covariance_gtsam + covariance_gtsam.T)
    information_gtsam = np.linalg.inv(covariance_gtsam)
    information_gtsam = 0.5 * (information_gtsam + information_gtsam.T)
    information_g2o = gtsam_information_to_g2o(information_gtsam)
    covariance_g2o = gtsam_information_to_g2o(covariance_gtsam)
    positive = eigenvalues[eigenvalues > floor]
    condition = (
        float(positive[-1] / positive[0]) if len(positive) >= 2 else float("inf")
    )
    return OrientedFactorResult(
        valid=True,
        reason="accepted",
        transform_target_source=transform,
        information_g2o=information_g2o,
        covariance_g2o=covariance_g2o,
        overlap=overlap,
        spatial_overlap=spatial_overlap,
        inlier_rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
        residual_scale_m=residual_scale,
        correspondence_count=len(source),
        orientation_consistent_count=oriented_count,
        observable_rank=observable_rank,
        scaled_information_eigenvalues=eigenvalues,
        scaled_condition_number=condition,
        iterations=iteration_count,
    )
