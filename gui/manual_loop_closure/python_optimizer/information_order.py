"""Explicit tangent-order conversions for anisotropic SE(3) information."""

from __future__ import annotations

import numpy as np


# G2O EDGE_SE3:QUAT: translation, rotation. GTSAM Pose3: rotation, translation.
G2O_TO_GTSAM = np.asarray([3, 4, 5, 0, 1, 2], dtype=np.int64)


def _validate_information(information: np.ndarray) -> np.ndarray:
    matrix = np.asarray(information, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError(f"SE(3) information must be 6x6, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("SE(3) information contains a non-finite value")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("SE(3) information must be symmetric")
    return matrix


def g2o_information_to_gtsam(information_g2o: np.ndarray) -> np.ndarray:
    matrix = _validate_information(information_g2o)
    return matrix[np.ix_(G2O_TO_GTSAM, G2O_TO_GTSAM)]


def gtsam_information_to_g2o(information_gtsam: np.ndarray) -> np.ndarray:
    # The block swap is its own inverse.
    matrix = _validate_information(information_gtsam)
    return matrix[np.ix_(G2O_TO_GTSAM, G2O_TO_GTSAM)]
