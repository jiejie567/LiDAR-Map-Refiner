"""Calibration and correlation controls for added loop-closure factors.

The input pose graph is left untouched.  These controls apply only to manual
or automatically proposed loop factors, whose uncertainty is otherwise easy
to over-count when several nearby keyframes observe the same physical place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CORRELATION_CLUSTERING_METHOD = "complete_link"


@dataclass(frozen=True)
class LoopFactorWeightInput:
    source_id: int
    target_id: int
    information: np.ndarray
    confidence: float = 1.0
    factor_scale: float = 1.0


@dataclass(frozen=True)
class LoopFactorWeightResult:
    cluster_id: int
    cluster_size: int
    correlation_scale: float
    global_scale: float
    applied_scale: float
    effective_information: np.ndarray


def _same_endpoint_cluster(
    left: LoopFactorWeightInput,
    right: LoopFactorWeightInput,
    window: int,
    policy: str,
) -> bool:
    direct = max(
        abs(int(left.source_id) - int(right.source_id)),
        abs(int(left.target_id) - int(right.target_id)),
    )
    swapped = max(
        abs(int(left.source_id) - int(right.target_id)),
        abs(int(left.target_id) - int(right.source_id)),
    )
    if policy == "pair":
        return min(direct, swapped) <= int(window)
    if policy == "shared_endpoint":
        endpoint_distance = min(
            abs(int(left.source_id) - int(right.source_id)),
            abs(int(left.source_id) - int(right.target_id)),
            abs(int(left.target_id) - int(right.source_id)),
            abs(int(left.target_id) - int(right.target_id)),
        )
        return endpoint_distance <= int(window)
    raise ValueError(f"unsupported correlation_policy: {policy!r}")


def _canonical_endpoints(item: LoopFactorWeightInput) -> tuple[int, int]:
    """Return an orientation-independent keyframe-pair coordinate."""
    source, target = int(item.source_id), int(item.target_id)
    return min(source, target), max(source, target)


def _clusters(
    inputs: list[LoopFactorWeightInput], window: int, policy: str,
) -> list[list[int]]:
    """Form deterministic, pairwise-compact endpoint neighbourhoods.

    Correlation proximity is not transitive: if A is near B and B is near C,
    A and C need not observe the same revisit.  Connected components therefore
    over-merge long chains and can suppress independent long-baseline loops.
    Greedy complete-link grouping keeps a factor in a cluster only when it is
    close to *every* member.  The canonical geometric ordering makes the
    result independent of proposal arrival order apart from exact ties.
    """
    if window <= 0:
        return [[index] for index in range(len(inputs))]
    ordered = sorted(
        range(len(inputs)),
        key=lambda index: (*_canonical_endpoints(inputs[index]), index),
    )
    grouped: list[list[int]] = []
    for index in ordered:
        for members in grouped:
            if all(
                _same_endpoint_cluster(
                    inputs[index], inputs[member], window, policy
                )
                for member in members
            ):
                members.append(index)
                break
        else:
            grouped.append([index])
    for members in grouped:
        members.sort()
    return sorted(grouped, key=lambda members: members[0])


def weight_loop_information(
    inputs: list[LoopFactorWeightInput],
    *,
    information_scale: float = 1.0,
    correlation_window_keyframes: int = 0,
    cluster_information_budget: float = 0.0,
    correlation_policy: str = "pair",
    cluster_allocation: str = "equal",
) -> list[LoopFactorWeightResult]:
    """Scale loop information while capping correlated evidence.

    ``cluster_information_budget`` is measured in effective independent loop
    factors.  A value of 1 assigns one factor's total information to every
    endpoint-neighbourhood cluster; zero disables the cap.  Relative
    confidence only allocates the cluster budget and never increases an
    individual factor above its original information.
    """
    if not np.isfinite(information_scale) or information_scale <= 0.0:
        raise ValueError("information_scale must be finite and positive")
    if correlation_window_keyframes < 0:
        raise ValueError("correlation_window_keyframes must be non-negative")
    if correlation_policy not in {"pair", "shared_endpoint"}:
        raise ValueError(
            "correlation_policy must be 'pair' or 'shared_endpoint'"
        )
    if cluster_allocation not in {"equal", "representative"}:
        raise ValueError(
            "cluster_allocation must be 'equal' or 'representative'"
        )
    if not np.isfinite(cluster_information_budget) or cluster_information_budget < 0.0:
        raise ValueError("cluster_information_budget must be finite and non-negative")
    for item in inputs:
        matrix = np.asarray(item.information, dtype=np.float64)
        if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
            raise ValueError("every loop information matrix must be finite and 6x6")
        if np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0] <= 0.0:
            raise ValueError("every loop information matrix must be SPD")
        if not np.isfinite(item.confidence) or item.confidence <= 0.0:
            raise ValueError("loop confidence must be finite and positive")
        if not np.isfinite(item.factor_scale) or item.factor_scale <= 0.0:
            raise ValueError("loop factor_scale must be finite and positive")

    results: list[LoopFactorWeightResult | None] = [None] * len(inputs)
    for cluster_id, members in enumerate(_clusters(
        inputs, correlation_window_keyframes, correlation_policy
    )):
        confidence_sum = sum(float(inputs[index].confidence) for index in members)
        representative = max(
            members,
            key=lambda index: (float(inputs[index].confidence), -index),
        )
        for index in members:
            if cluster_information_budget > 0.0:
                if cluster_allocation == "representative":
                    # Keep the first/highest-confidence observation as the
                    # cluster representative.  Redundant factors remain in the
                    # graph and ledger with negligible numerical influence,
                    # avoiding a discontinuous reweighting of older factors
                    # whenever a new correlated loop arrives.
                    allocated = (
                        float(cluster_information_budget)
                        if index == representative else 1e-9
                    )
                else:
                    allocated = (
                        float(cluster_information_budget)
                        * float(inputs[index].confidence)
                        / confidence_sum
                    )
                correlation_scale = min(1.0, allocated)
            else:
                correlation_scale = 1.0
            applied = (
                float(information_scale)
                * float(inputs[index].factor_scale)
                * correlation_scale
            )
            results[index] = LoopFactorWeightResult(
                cluster_id=cluster_id,
                cluster_size=len(members),
                correlation_scale=correlation_scale,
                global_scale=float(information_scale),
                applied_scale=applied,
                effective_information=(
                    np.asarray(inputs[index].information, dtype=np.float64) * applied
                ),
            )
    return [result for result in results if result is not None]
