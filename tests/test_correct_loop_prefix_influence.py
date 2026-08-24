from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/ghostloop_eval/analyze_correct_loop_prefix_influence.py"
)
SPEC = importlib.util.spec_from_file_location("correct_loop_influence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def pose(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    result[0, 3] = x
    return result


def edge(node_i: int, node_j: int, information: float = 1000.0):
    return MODULE.EdgeRecord(
        node_i=node_i,
        node_j=node_j,
        measurement=np.linalg.inv(pose(float(node_i))) @ pose(float(node_j)),
        information_g2o=np.eye(6) * information,
    )


def test_pose_residual_is_zero_for_matching_measurement() -> None:
    left, right = pose(1.0, 10.0), pose(3.0, 15.0)
    translation, rotation = MODULE.pose_residual(
        left, right, np.linalg.inv(left) @ right
    )
    np.testing.assert_allclose(translation, 0.0, atol=1e-12)
    np.testing.assert_allclose(rotation, 0.0, atol=1e-12)


def test_trajectory_motion_reports_translation_and_rotation() -> None:
    before = np.stack([pose(), pose(1.0)])
    after = np.stack([pose(), pose(1.2, 10.0)])
    result = MODULE.trajectory_motion(before, after)
    assert result["trajectory_translation_delta_m_max"] == pytest.approx(0.2)
    assert result["trajectory_rotation_delta_deg_max"] == pytest.approx(10.0)
    assert result["trajectory_translation_delta_m_median"] == pytest.approx(0.1)


def test_chain_channel_leverage_exposes_rotation_heavy_loop() -> None:
    edges = [edge(0, 1), edge(1, 2), edge(2, 3)]
    # g2o channel order is translation then rotation.  The loop has weak
    # translation information and strong rotation information.
    loop_information = np.diag([0.0625] * 3 + [1313.0] * 3)
    result = MODULE.chain_channel_leverage(3, 0, edges, loop_information)
    assert result["translation_channel_gain"] == pytest.approx(
        0.003 / (0.003 + 16.0)
    )
    assert result["rotation_channel_gain"] == pytest.approx(
        0.003 / (0.003 + 1.0 / 1313.0)
    )
    assert result["rotation_to_translation_gain_ratio"] > 4000.0


def test_chain_channel_variance_adds_adjacent_edge_covariances() -> None:
    result = MODULE.chain_channel_variances(
        3, 0, [edge(0, 1), edge(1, 2), edge(2, 3)]
    )
    assert result is not None
    assert result[0] == pytest.approx(0.003)
    assert result[1] == pytest.approx(0.003)


def test_auc_is_one_for_perfect_high_value_predictor() -> None:
    assert MODULE._auc([0.0, 1.0, 2.0, 3.0], [False, False, True, True]) == 1.0
