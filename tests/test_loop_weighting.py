from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


GUI = Path(__file__).resolve().parents[1] / "gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from manual_loop_closure.python_optimizer.loop_weighting import (  # noqa: E402
    LoopFactorWeightInput,
    weight_loop_information,
)
from manual_loop_closure.python_optimizer.optimizer import (  # noqa: E402
    ManualConstraintSpec,
    _manual_constraint_information,
)
from manual_loop_closure.repair_presets import (  # noqa: E402
    PRODUCTION_PGO_PROFILE,
    REPAIR_ENV_PRESETS,
    production_loop_variances,
)


def test_production_pgo_profile_is_small_and_unit_consistent() -> None:
    variance_t, variance_r = production_loop_variances()
    assert PRODUCTION_PGO_PROFILE["optimize_mode"] == "gnc_tls"
    assert PRODUCTION_PGO_PROFILE["manual_information_scale"] == pytest.approx(0.1)
    assert PRODUCTION_PGO_PROFILE["correlation_window_keyframes"] == 0
    assert PRODUCTION_PGO_PROFILE["cluster_information_budget"] == 0.0
    assert variance_t == pytest.approx((1.6, 1.6, 1.6))
    assert np.degrees(np.sqrt(variance_r[0])) == pytest.approx(0.5)


def test_production_balm_uses_one_convergence_cap_in_all_environments() -> None:
    assert REPAIR_ENV_PRESETS["indoor"]["balm_iterations"] == 20
    assert REPAIR_ENV_PRESETS["outdoor"]["balm_iterations"] == 20


def item(
    source: int, target: int, confidence: float = 1.0,
    factor_scale: float = 1.0,
):
    return LoopFactorWeightInput(
        source_id=source,
        target_id=target,
        information=np.eye(6),
        confidence=confidence,
        factor_scale=factor_scale,
    )


def constraint(unit: str, sigma: float) -> ManualConstraintSpec:
    return ManualConstraintSpec(
        enabled=True,
        source_id=10,
        target_id=0,
        translation_xyz=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        sigma_t_xyz=np.ones(3),
        sigma_r_deg=np.full(3, sigma),
        sigma_rotation_unit=unit,
    )


def test_rotation_units_are_explicit_and_equivalent() -> None:
    degrees = _manual_constraint_information(constraint("deg", 180.0 / np.pi))
    radians = _manual_constraint_information(constraint("rad", 1.0))
    np.testing.assert_allclose(degrees, radians)


def test_unknown_rotation_unit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="sigma_rotation_unit"):
        _manual_constraint_information(constraint("grad", 1.0))


def test_nearby_endpoint_loops_share_one_information_budget() -> None:
    results = weight_loop_information(
        [item(100, 10), item(104, 13), item(300, 20)],
        correlation_window_keyframes=5,
        cluster_information_budget=1.0,
    )
    assert [result.cluster_size for result in results] == [2, 2, 1]
    assert [result.correlation_scale for result in results] == [0.5, 0.5, 1.0]
    np.testing.assert_allclose(results[0].effective_information, np.eye(6) * 0.5)


def test_swapped_endpoints_cluster_and_confidence_allocates_budget() -> None:
    results = weight_loop_information(
        [item(100, 10, 3.0), item(12, 102, 1.0)],
        correlation_window_keyframes=2,
        cluster_information_budget=1.0,
    )
    assert results[0].cluster_id == results[1].cluster_id
    assert results[0].correlation_scale == pytest.approx(0.75)
    assert results[1].correlation_scale == pytest.approx(0.25)


def test_endpoint_proximity_does_not_chain_independent_loops() -> None:
    results = weight_loop_information(
        [item(100, 10), item(105, 15), item(110, 20)],
        correlation_window_keyframes=5,
        cluster_information_budget=1.0,
    )
    # The middle factor is close to both neighbours, but the two outer pairs
    # are ten keyframes apart and must not be merged transitively.
    assert [result.cluster_size for result in results] == [2, 2, 1]
    assert [result.cluster_id for result in results] == [0, 0, 1]
    assert [result.correlation_scale for result in results] == [0.5, 0.5, 1.0]


def test_global_scale_and_budget_compose_without_upweighting_member() -> None:
    results = weight_loop_information(
        [item(100, 10), item(101, 11)],
        information_scale=0.2,
        correlation_window_keyframes=2,
        cluster_information_budget=4.0,
    )
    assert [result.correlation_scale for result in results] == [1.0, 1.0]
    np.testing.assert_allclose(results[0].effective_information, np.eye(6) * 0.2)


def test_per_factor_scale_composes_with_global_scale() -> None:
    results = weight_loop_information(
        [item(100, 10, factor_scale=0.25)], information_scale=0.4
    )
    assert results[0].applied_scale == pytest.approx(0.1)
    np.testing.assert_allclose(results[0].effective_information, np.eye(6) * 0.1)


def test_shared_endpoint_policy_clusters_reused_trajectory_segment() -> None:
    pair_policy = weight_loop_information(
        [item(554, 121), item(443, 115)],
        correlation_window_keyframes=30,
        cluster_information_budget=1.0,
        correlation_policy="pair",
    )
    shared_policy = weight_loop_information(
        [item(554, 121), item(443, 115)],
        correlation_window_keyframes=30,
        cluster_information_budget=1.0,
        correlation_policy="shared_endpoint",
    )
    assert [result.cluster_size for result in pair_policy] == [1, 1]
    assert [result.cluster_size for result in shared_policy] == [2, 2]
    assert [result.correlation_scale for result in shared_policy] == [0.5, 0.5]


def test_representative_allocation_does_not_reweight_first_factor() -> None:
    results = weight_loop_information(
        [item(554, 121), item(443, 115)],
        correlation_window_keyframes=30,
        cluster_information_budget=1.0,
        correlation_policy="shared_endpoint",
        cluster_allocation="representative",
    )
    assert results[0].correlation_scale == 1.0
    assert results[1].correlation_scale == pytest.approx(1e-9)
