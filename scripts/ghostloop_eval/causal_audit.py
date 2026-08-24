"""Candidate-local causal audit for map-inconsistency loop proposals.

The detector is allowed to propose a hypothesis, but it is not allowed to
change the evidence while that hypothesis is evaluated.  This module freezes
the exact local points, observing side, per-voxel layer labels, and surface
normals at proposal time, then reprojects those same observations under a
trial pose-graph solution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

try:
    from manual_loop_closure.python_optimizer.balm import (
        _canonical_plane_normal,
        _two_means_1d,
    )
except ImportError:  # Direct script/test imports add python_optimizer itself.
    from balm import _canonical_plane_normal, _two_means_1d


DEFAULT_CLOSE_THRESHOLD_M = 0.08
DEFAULT_MIN_ABSOLUTE_DROP_M = 0.03
DEFAULT_MIN_RELATIVE_DROP = 0.30
DEFAULT_FACTOR_TAIL_PROBABILITY = 0.9973
DEFAULT_GRAVITY_MAX_ERROR_DEG = 1.0


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def se3_log_g2o(transform: np.ndarray) -> np.ndarray:
    """SE(3) logarithm in the g2o [translation, rotation] tangent order."""
    transform = np.asarray(transform, dtype=np.float64)
    omega = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-6:
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega_hat + omega_hat @ omega_hat / 12.0
        )
    else:
        coefficient = (
            1.0 / theta**2
            - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega_hat
            + coefficient * (omega_hat @ omega_hat)
        )
    return np.concatenate([
        inverse_left_jacobian @ transform[:3, 3], omega
    ])


@dataclass(frozen=True)
class FactorConsistency:
    translation_residual_m: float
    rotation_residual_deg: float
    nis: float
    chi_square_threshold: float
    normalized_score: float


@dataclass(frozen=True)
class GravityConsistency:
    available: bool
    error_deg: float | None
    max_error_deg: float
    passed: bool
    reason: str


def gravity_consistency(
    measured_target_source: np.ndarray,
    source_up_local: np.ndarray | None,
    target_up_local: np.ndarray | None,
    *,
    max_error_deg: float = DEFAULT_GRAVITY_MAX_ERROR_DEG,
) -> GravityConsistency:
    """Check the one rotational component an ICP factor cannot negotiate.

    ``measured_target_source`` maps source-frame vectors into the target frame,
    so a physically valid factor must also map the measured source up-vector
    onto the target up-vector.  This check is independent of global frame and
    yaw, and therefore remains valid for descriptor- and map-proposed loops.
    """
    threshold = float(max_error_deg)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("max_error_deg must be finite and positive")
    if source_up_local is None or target_up_local is None:
        return GravityConsistency(
            False, None, threshold, True, "gravity sidecar unavailable"
        )
    source = np.asarray(source_up_local, dtype=np.float64).reshape(3)
    target = np.asarray(target_up_local, dtype=np.float64).reshape(3)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if (
        not np.all(np.isfinite(source)) or not np.all(np.isfinite(target))
        or source_norm < 1e-9 or target_norm < 1e-9
    ):
        return GravityConsistency(
            False, None, threshold, True, "gravity vectors invalid"
        )
    rotation = np.asarray(measured_target_source, dtype=np.float64)[:3, :3]
    predicted_target = rotation @ (source / source_norm)
    cosine = float(np.clip(
        np.dot(predicted_target, target / target_norm), -1.0, 1.0
    ))
    error_deg = float(np.degrees(np.arccos(cosine)))
    passed = error_deg <= threshold
    return GravityConsistency(
        True,
        error_deg,
        threshold,
        passed,
        (
            "gravity direction is consistent"
            if passed else
            f"gravity error {error_deg:.3f} deg exceeds {threshold:.3f} deg"
        ),
    )


def factor_consistency(
    measured_target_source: np.ndarray,
    solved_target_source: np.ndarray,
    information_g2o: np.ndarray,
    *,
    tail_probability: float = DEFAULT_FACTOR_TAIL_PROBABILITY,
) -> FactorConsistency:
    """Evaluate a solved factor using its own calibrated uncertainty.

    A scalar translation cutoff treats a wall-normal constraint and a
    tangentially unobservable constraint as equally certain.  The full factor
    information instead yields a chi-square normalized innovation score; one
    means the residual is exactly at the preregistered tail threshold.
    """
    measured = np.asarray(measured_target_source, dtype=np.float64)
    solved = np.asarray(solved_target_source, dtype=np.float64)
    information = np.asarray(information_g2o, dtype=np.float64)
    if information.shape != (6, 6):
        raise ValueError("information_g2o must be 6x6")
    # The estimator covariance is built for a left perturbation in target frame.
    error = se3_log_g2o(solved @ np.linalg.inv(measured))
    nis = max(0.0, float(error @ information @ error))
    rank = max(1, int(np.linalg.matrix_rank(information)))
    threshold = float(chi2.ppf(tail_probability, rank))
    return FactorConsistency(
        translation_residual_m=float(np.linalg.norm(error[:3])),
        rotation_residual_deg=float(np.degrees(np.linalg.norm(error[3:]))),
        nis=nis,
        chi_square_threshold=threshold,
        normalized_score=nis / max(threshold, 1e-12),
    )


@dataclass(frozen=True)
class GraphTrialEffect:
    candidate_residual_m: float
    worst_factor_residual_m: float
    base_worst_factor_residual_m: float
    retained: bool
    reason: str


def evaluate_graph_trial(
    candidate_residual_m: float,
    worst_factor_residual_m: float,
    base_worst_factor_residual_m: float,
    *,
    residual_limit_m: float,
) -> GraphTrialEffect:
    """Apply the shared trial-PGO factor-consistency gate."""
    candidate = float(candidate_residual_m)
    worst = float(worst_factor_residual_m)
    base_worst = float(base_worst_factor_residual_m)
    candidate_ok = candidate <= residual_limit_m
    newly_global = (
        worst > residual_limit_m
        and worst > max(residual_limit_m, base_worst * 1.05)
    )
    if not candidate_ok:
        retained, reason = False, "candidate factor exceeds 3-sigma residual"
    elif newly_global:
        retained, reason = False, "trial introduces a new global residual violation"
    else:
        retained, reason = True, "trial graph passes factor consistency"
    return GraphTrialEffect(candidate, worst, base_worst, retained, reason)


def _key_tuple(key: Sequence[int]) -> tuple[int, int, int]:
    return tuple(int(value) for value in key)


def _interval_fraction(ids: np.ndarray, interval: Sequence[int]) -> float:
    if not len(ids):
        return 0.0
    return float(np.mean((ids >= int(interval[0])) & (ids <= int(interval[1]))))


@dataclass(frozen=True)
class FrozenGhostEvidence:
    """Immutable observations used to test one map-inconsistency hypothesis."""

    local_xyz: np.ndarray
    pose_ids: np.ndarray
    layer_low: np.ndarray
    voxel_ids: np.ndarray
    voxel_normals: np.ndarray
    voxel_keys: np.ndarray
    reference_separation_m: float
    reported_separation_m: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            local_xyz=self.local_xyz,
            pose_ids=self.pose_ids,
            layer_low=self.layer_low,
            voxel_ids=self.voxel_ids,
            voxel_normals=self.voxel_normals,
            voxel_keys=self.voxel_keys,
            reference_separation_m=np.asarray(self.reference_separation_m),
            reported_separation_m=np.asarray(self.reported_separation_m),
        )

    @classmethod
    def load(cls, path: Path) -> "FrozenGhostEvidence":
        with np.load(path) as data:
            return cls(
                local_xyz=data["local_xyz"],
                pose_ids=data["pose_ids"],
                layer_low=data["layer_low"],
                voxel_ids=data["voxel_ids"],
                voxel_normals=data["voxel_normals"],
                voxel_keys=data["voxel_keys"],
                reference_separation_m=float(data["reference_separation_m"]),
                reported_separation_m=float(data["reported_separation_m"]),
            )


@dataclass(frozen=True)
class CausalEffect:
    before_m: float
    after_m: float
    absolute_drop_m: float
    relative_drop: float
    retained: bool
    reason: str


def evaluate_causal_effect(
    before_m: float,
    after_m: float,
    *,
    close_threshold_m: float = DEFAULT_CLOSE_THRESHOLD_M,
    min_absolute_drop_m: float = DEFAULT_MIN_ABSOLUTE_DROP_M,
    min_relative_drop: float = DEFAULT_MIN_RELATIVE_DROP,
) -> CausalEffect:
    """Apply the preregistered local-effect gate from the ICRA protocol."""
    before = float(before_m)
    after = float(after_m)
    absolute = before - after
    relative = absolute / max(before, 1e-12)
    closed = after < close_threshold_m
    substantial = absolute >= min_absolute_drop_m and relative >= min_relative_drop
    if closed:
        retained, reason = True, "fixed evidence closed below 8 cm"
    elif substantial:
        retained, reason = True, "fixed evidence dropped by >=3 cm and >=30%"
    elif after > before + 1e-9:
        retained, reason = False, "fixed evidence worsened"
    else:
        retained, reason = False, "fixed evidence did not improve enough"
    return CausalEffect(before, after, absolute, relative, retained, reason)


def _select_layers(
    region: dict,
    voxel_key: tuple[int, int, int],
    local: np.ndarray,
    pose_ids: np.ndarray,
    poses: np.ndarray,
    component: dict | None,
) -> dict:
    transforms = poses[pose_ids]
    world = (
        np.einsum("nij,nj->ni", transforms[:, :3, :3], local)
        + transforms[:, :3, 3]
    )
    sensors = transforms[:, :3, 3]
    mean = world.mean(axis=0)
    centered = world - mean
    if component is None:
        _, eigenvectors = np.linalg.eigh(centered.T @ centered / len(world))
        normal = _canonical_plane_normal(eigenvectors[:, 0])
        expected_side = None
        expected_separation = float(region["separation_typical_m"])
    else:
        normal = np.asarray(component["normal"], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        expected_side = int(component["view_side"])
        expected_separation = float(component["separation_m"])
    view_dot = np.einsum("ij,j->i", sensors - world, normal)
    range_a = region["layer_a_keyframes"]
    range_b = region["layer_b_keyframes"]
    options = (
        (view_dot >= 0.0,) if expected_side == 1
        else (view_dot < 0.0,) if expected_side == -1
        else (view_dot >= 0.0, view_dot < 0.0)
    )
    candidates = []
    for side_mask in options:
        if int(side_mask.sum()) < 40:
            continue
        selected = np.flatnonzero(side_mask)
        offsets = (world[selected] - mean) @ normal
        split = _two_means_1d(offsets)
        if split is None:
            continue
        low_center, high_center, layer_low = split
        low_ids = pose_ids[selected[layer_low]]
        high_ids = pose_ids[selected[~layer_low]]
        direct = min(
            _interval_fraction(low_ids, range_a),
            _interval_fraction(high_ids, range_b),
        )
        crossed = min(
            _interval_fraction(low_ids, range_b),
            _interval_fraction(high_ids, range_a),
        )
        separation = float(high_center - low_center)
        score = max(direct, crossed) - 0.25 * abs(separation - expected_separation)
        candidates.append((score, selected, layer_low, normal.copy(), separation))
    if not candidates:
        raise RuntimeError(f"No same-side layer split for voxel {voxel_key}")
    _, selected, layer_low, normal, separation = max(candidates, key=lambda item: item[0])
    return {
        "local": local[selected],
        "pose_ids": pose_ids[selected],
        "layer_low": layer_low,
        "normal": normal,
        "reference_separation_m": separation,
    }


def freeze_region_evidence(
    region: dict,
    local_clouds: Iterable[np.ndarray],
    poses_world_sensor: np.ndarray,
    *,
    root_voxel_size: float,
    max_observation_range: float,
    report_tolerance_m: float = 0.01,
) -> FrozenGhostEvidence:
    """Freeze all detector observations supporting one connected region."""
    poses = np.asarray(poses_world_sensor, dtype=float)
    selected_keys = {_key_tuple(key) for key in region["voxel_keys"]}
    local_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    pose_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for frame_index, raw_local in enumerate(local_clouds):
        local = np.asarray(raw_local, dtype=float)
        if local.size == 0:
            continue
        transform = poses[frame_index]
        world = local @ transform[:3, :3].T + transform[:3, 3]
        near = np.ones(len(local), dtype=bool)
        if max_observation_range > 0.0:
            delta = world - transform[:3, 3]
            near = np.einsum("ij,ij->i", delta, delta) <= max_observation_range**2
        local = local[near]
        keys = np.floor(world[near] / float(root_voxel_size)).astype(np.int64)
        for key in selected_keys:
            mask = np.all(keys == np.asarray(key, dtype=np.int64), axis=1)
            if np.any(mask):
                local_by_key[key].append(local[mask])
                pose_by_key[key].append(
                    np.full(int(mask.sum()), frame_index, dtype=np.int32)
                )

    component_by_key = {
        _key_tuple(item["voxel_key"]): item
        for item in region.get("component_voxels", [])
    }
    groups = []
    for key_value in region["voxel_keys"]:
        key = _key_tuple(key_value)
        if key not in local_by_key:
            raise RuntimeError(f"No observations for candidate voxel {key}")
        groups.append(
            _select_layers(
                region,
                key,
                np.vstack(local_by_key[key]),
                np.concatenate(pose_by_key[key]),
                poses,
                component_by_key.get(key),
            )
        )
    evidence = FrozenGhostEvidence(
        local_xyz=np.vstack([group["local"] for group in groups]),
        pose_ids=np.concatenate([group["pose_ids"] for group in groups]),
        layer_low=np.concatenate([group["layer_low"] for group in groups]),
        voxel_ids=np.concatenate([
            np.full(len(group["local"]), index, dtype=np.int32)
            for index, group in enumerate(groups)
        ]),
        voxel_normals=np.vstack([group["normal"] for group in groups]),
        voxel_keys=np.asarray(region["voxel_keys"], dtype=np.int64),
        reference_separation_m=0.0,
        reported_separation_m=float(region["separation_typical_m"]),
    )
    measured = measure_frozen_evidence(evidence, poses)
    if abs(measured - evidence.reported_separation_m) > report_tolerance_m:
        raise RuntimeError(
            f"Fixed separation {measured:.3f} m differs from detector typical "
            f"{evidence.reported_separation_m:.3f} m"
        )
    return FrozenGhostEvidence(
        local_xyz=evidence.local_xyz,
        pose_ids=evidence.pose_ids,
        layer_low=evidence.layer_low,
        voxel_ids=evidence.voxel_ids,
        voxel_normals=evidence.voxel_normals,
        voxel_keys=evidence.voxel_keys,
        reference_separation_m=measured,
        reported_separation_m=evidence.reported_separation_m,
    )


def measure_frozen_evidence(
    evidence: FrozenGhostEvidence,
    poses_world_sensor: np.ndarray,
) -> float:
    """Measure weighted layer separation without re-detection or relabeling."""
    poses = np.asarray(poses_world_sensor, dtype=float)
    transforms = poses[evidence.pose_ids]
    world = (
        np.einsum("nij,nj->ni", transforms[:, :3, :3], evidence.local_xyz)
        + transforms[:, :3, 3]
    )
    separations = []
    weights = []
    for voxel_id, normal in enumerate(evidence.voxel_normals):
        selected = evidence.voxel_ids == voxel_id
        labels = evidence.layer_low[selected]
        if not np.any(labels) or not np.any(~labels):
            continue
        offsets = world[selected] @ normal
        separations.append(abs(float(offsets[labels].mean() - offsets[~labels].mean())))
        weights.append(int(selected.sum()))
    if not separations:
        raise RuntimeError("Frozen evidence contains no measurable two-layer voxel")
    return float(np.average(separations, weights=weights))
