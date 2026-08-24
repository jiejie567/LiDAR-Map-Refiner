from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure.python_optimizer.information_order import (  # noqa: E402
    g2o_information_to_gtsam,
    gtsam_information_to_g2o,
)
from manual_loop_closure.python_optimizer.oriented_surface_factors import (  # noqa: E402
    OrientedFactorConfig,
    estimate_oriented_surface_factor,
)


class InformationOrderTest(unittest.TestCase):
    def test_anisotropic_information_round_trip_and_block_swap(self):
        matrix = np.diag([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        matrix[0, 4] = matrix[4, 0] = 0.25
        converted = g2o_information_to_gtsam(matrix)
        np.testing.assert_allclose(
            np.diag(converted), [10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        )
        self.assertEqual(converted[3, 1], 0.25)
        np.testing.assert_allclose(gtsam_information_to_g2o(converted), matrix)

    def test_rejects_non_symmetric_information(self):
        matrix = np.eye(6)
        matrix[0, 1] = 1.0
        with self.assertRaises(ValueError):
            g2o_information_to_gtsam(matrix)


class OrientedSurfaceFactorTest(unittest.TestCase):
    @staticmethod
    def _room(seed: int = 4) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = 900
        floor = np.column_stack([
            rng.uniform(-4, 4, n), rng.uniform(-3, 3, n),
            rng.normal(-1.2, 0.004, n),
        ])
        wall_x = np.column_stack([
            rng.normal(4.0, 0.004, n), rng.uniform(-3, 3, n),
            rng.uniform(-1.0, 2.0, n),
        ])
        wall_y = np.column_stack([
            rng.uniform(-4, 4, n), rng.normal(3.0, 0.004, n),
            rng.uniform(-1.0, 2.0, n),
        ])
        return np.vstack([floor, wall_x, wall_y])

    def test_recovers_transform_and_returns_full_spd_information(self):
        target = self._room()
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [0.01, -0.015, 0.025]
        ).as_matrix()
        expected[:3, 3] = [0.12, -0.08, 0.04]
        source = (target - expected[:3, 3]) @ expected[:3, :3]
        initial = np.eye(4)
        config = OrientedFactorConfig(
            voxel_size_m=0.18, normal_radius_m=0.65,
            max_correspondence_m=0.5, min_correspondences=100,
            min_overlap=0.5,
        )
        result = estimate_oriented_surface_factor(source, target, initial, config)
        self.assertTrue(result.valid, result.reason)
        gap = np.linalg.inv(result.transform_target_source) @ expected
        self.assertLess(np.linalg.norm(gap[:3, 3]), 0.03)
        self.assertLess(Rotation.from_matrix(gap[:3, :3]).magnitude(), 0.02)
        np.testing.assert_allclose(
            result.information_g2o, result.information_g2o.T, atol=1e-8
        )
        self.assertGreater(np.linalg.eigvalsh(result.information_g2o).min(), 0.0)
        self.assertGreaterEqual(result.observable_rank, 5)

    def test_opposite_observation_sides_do_not_form_a_factor(self):
        rng = np.random.default_rng(8)
        n = 1800
        # Identical geometric sheet, but each local cloud observes it from the
        # opposite side. Normals oriented to the local sensor are antiparallel.
        yz = np.column_stack([
            rng.uniform(-3, 3, n), rng.uniform(-1.5, 1.5, n)
        ])
        target = np.column_stack([np.full(n, 2.0), yz])
        source = np.column_stack([np.full(n, -2.0), yz])
        initial = np.eye(4)
        initial[0, 3] = 4.0
        result = estimate_oriented_surface_factor(
            source, target, initial,
            OrientedFactorConfig(
                voxel_size_m=0.15, normal_radius_m=0.5,
                max_correspondence_m=0.25, normal_gate_deg=30,
                min_correspondences=80, min_overlap=0.5,
            ),
        )
        self.assertFalse(result.valid)
        self.assertIn(result.reason, {
            "too_few_oriented_correspondences", "insufficient_final_support"
        })

    def test_oriented_search_avoids_nearer_opposite_wall_face(self):
        rng = np.random.default_rng(31)
        count = 1800
        yz = np.column_stack([
            rng.uniform(-3.0, 3.0, count),
            rng.uniform(-1.5, 1.5, count),
        ])
        # The source observes the x=0 face from its negative side.  The target
        # submap contains that face and the opposite face of a 12 cm partition.
        source = np.column_stack([rng.normal(0.0, 0.002, count), yz])
        target_same = np.column_stack([rng.normal(0.0, 0.002, count), yz])
        target_opposite = np.column_stack([
            rng.normal(0.12, 0.002, count), yz
        ])
        target = np.vstack([target_same, target_opposite])
        source_normals = np.tile([-1.0, 0.0, 0.0], (count, 1))
        target_normals = np.vstack([
            np.tile([-1.0, 0.0, 0.0], (count, 1)),
            np.tile([1.0, 0.0, 0.0], (count, 1)),
        ])
        initial = np.eye(4)
        initial[0, 3] = 0.09  # spatially closer to the wrong x=0.12 face
        common = dict(
            voxel_size_m=0.04, max_correspondence_m=0.25,
            min_correspondences=100, min_overlap=0.5,
            normal_radius_m=0.2,
        )
        unoriented = estimate_oriented_surface_factor(
            source, target, initial,
            OrientedFactorConfig(
                **common, normal_gate_deg=180.0, normal_search_k=1,
            ),
            source_normals_local=source_normals,
            target_normals_local=target_normals,
        )
        oriented = estimate_oriented_surface_factor(
            source, target, initial,
            OrientedFactorConfig(
                **common, normal_gate_deg=30.0, normal_search_k=64,
            ),
            source_normals_local=source_normals,
            target_normals_local=target_normals,
        )
        self.assertTrue(unoriented.valid, unoriented.reason)
        self.assertTrue(oriented.valid, oriented.reason)
        self.assertGreater(unoriented.transform_target_source[0, 3], 0.08)
        self.assertLess(abs(oriented.transform_target_source[0, 3]), 0.02)

    def test_corridor_exposes_a_weak_direction(self):
        rng = np.random.default_rng(12)
        n = 1600
        left = np.column_stack([
            rng.uniform(-10, 10, n), rng.normal(-2.0, 0.004, n),
            rng.uniform(-1, 2, n),
        ])
        right = np.column_stack([
            rng.uniform(-10, 10, n), rng.normal(2.0, 0.004, n),
            rng.uniform(-1, 2, n),
        ])
        floor = np.column_stack([
            rng.uniform(-10, 10, n), rng.uniform(-2, 2, n),
            rng.normal(-1.0, 0.004, n),
        ])
        target = np.vstack([left, right, floor])
        source = target.copy()
        result = estimate_oriented_surface_factor(
            source, target, np.eye(4),
            OrientedFactorConfig(
                voxel_size_m=0.22, normal_radius_m=0.7,
                max_correspondence_m=0.4, min_correspondences=100,
                min_overlap=0.7, degeneracy_ratio=2e-2,
            ),
        )
        self.assertTrue(result.valid, result.reason)
        self.assertLess(result.observable_rank, 6)
        self.assertLess(
            result.scaled_information_eigenvalues[0],
            2e-2 * result.scaled_information_eigenvalues[-1],
        )

    def test_bias_floor_caps_factor_confidence_in_physical_units(self):
        target = self._room(seed=41)
        source = target.copy()
        common = dict(
            voxel_size_m=0.18, normal_radius_m=0.65,
            max_correspondence_m=0.4, min_correspondences=100,
            min_overlap=0.7,
        )
        baseline = estimate_oriented_surface_factor(
            source, target, np.eye(4), OrientedFactorConfig(**common)
        )
        floored = estimate_oriented_surface_factor(
            source, target, np.eye(4),
            OrientedFactorConfig(
                **common,
                translation_sigma_floor_m=0.20,
                rotation_sigma_floor_rad=0.03,
            ),
        )
        self.assertTrue(baseline.valid, baseline.reason)
        self.assertTrue(floored.valid, floored.reason)
        covariance_delta = (
            g2o_information_to_gtsam(floored.covariance_g2o)
            - g2o_information_to_gtsam(baseline.covariance_g2o)
        )
        np.testing.assert_allclose(
            covariance_delta,
            np.diag([0.03**2] * 3 + [0.20**2] * 3),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
