from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import manual_loop_closure.registration as registration  # noqa: E402


def _plane() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 7), np.linspace(-1.0, 1.0, 7))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


class ObservationOrientedIcpTest(unittest.TestCase):
    def test_production_registration_keeps_m3_disabled(self) -> None:
        config = registration.RegistrationConfig()

        self.assertFalse(config.normal_gate)
        self.assertAlmostEqual(config.normal_gate_max_angle_deg, 60.0)
        self.assertAlmostEqual(config.normal_radius, 0.25)

    def test_signed_gate_rejects_oppositely_observed_faces(self) -> None:
        points = _plane()
        source_normals = np.repeat([[0.0, 0.0, 1.0]], points.shape[0], axis=0)
        target_normals = -source_normals
        prepared = registration.GatedIcpInputs(
            source=points,
            source_normals=source_normals,
            target=points.copy(),
            target_normals=target_normals,
            tree=cKDTree(points),
        )

        transform, diagnostics = registration.registration_normal_gated_icp(
            points,
            np.zeros_like(points),
            points,
            np.zeros_like(points),
            np.eye(4),
            voxel_size=0.1,
            max_correspondence_distance=0.5,
            max_iterations=5,
            normal_radius=0.25,
            max_angle_deg=60.0,
            prepared=prepared,
        )

        np.testing.assert_allclose(transform, np.eye(4))
        self.assertEqual(diagnostics.spatial_correspondence_count, points.shape[0])
        self.assertEqual(diagnostics.oriented_correspondence_count, 0)
        self.assertFalse(diagnostics.valid)
        self.assertEqual(
            diagnostics.termination_reason,
            "insufficient_oriented_correspondences",
        )

    def test_m3_refines_point_to_plane_inside_plain_gicp_basin(self) -> None:
        source = _plane()
        target = source + np.array([0.0, 0.0, 0.2])
        normals = np.repeat([[0.0, 0.0, 1.0]], source.shape[0], axis=0)
        prepared = registration.GatedIcpInputs(
            source=source,
            source_normals=normals,
            target=target,
            target_normals=normals.copy(),
            tree=cKDTree(target),
        )

        transform, diagnostics = registration.registration_normal_gated_icp(
            source,
            np.zeros_like(source),
            target,
            np.zeros_like(target),
            np.eye(4),
            voxel_size=0.1,
            max_correspondence_distance=0.5,
            max_iterations=10,
            normal_radius=0.25,
            max_angle_deg=60.0,
            prepared=prepared,
        )

        self.assertAlmostEqual(float(transform[2, 3]), 0.2, places=5)
        self.assertTrue(diagnostics.valid)
        self.assertEqual(
            diagnostics.oriented_correspondence_count,
            source.shape[0],
        )
        self.assertLess(diagnostics.oriented_rmse, 1e-5)

    def test_normal_cache_evicts_old_frames_at_point_budget(self) -> None:
        workspace = registration.RegistrationWorkspace(
            Path("/unused"),
            SimpleNamespace(),
        )
        points = np.zeros((2, 3), dtype=np.float64)
        workspace.load_local_points = mock.Mock(return_value=points)
        cache: OrderedDict = OrderedDict()
        with (
            mock.patch.object(registration, "_FRAME_NORMALS_CACHE", cache),
            mock.patch.object(registration, "_normal_cache_points_total", 0),
            mock.patch.object(registration, "NORMAL_CACHE_MAX_POINTS", 3),
            mock.patch.object(
                registration,
                "oriented_normals",
                return_value=np.ones((2, 3), dtype=np.float64),
            ),
        ):
            workspace.local_oriented_normals(1, 0.25)
            workspace.local_oriented_normals(2, 0.25)

            self.assertEqual(len(cache), 1)
            self.assertEqual(next(iter(cache))[1], 2)
            self.assertEqual(registration._normal_cache_points_total, 2)


if __name__ == "__main__":
    unittest.main()
