from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "scripts" / "oriented_surface"
)
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_oriented_pose_graph import (  # noqa: E402
    edge_line,
    format_information,
    isotropize_information,
    load_loop_rows,
    parse_g2o_edge,
    parse_information,
    REWEIGHT_VARIANTS,
    LOOP_ONLY_VARIANTS,
)
from calibrate_factor_information import (  # noqa: E402
    calibration_scale,
    factor_error_g2o,
    interpolated_truth,
    se3_log_g2o,
)
from generate_proximity_loop_candidates import proximity_pairs  # noqa: E402


class OrientedGraphRebuildTest(unittest.TestCase):
    def test_reweight_variants_are_explicit_and_do_not_alias_replacement(self):
        self.assertEqual(
            REWEIGHT_VARIANTS,
            {"icp_anisotropic_reweight", "oriented_anisotropic_reweight"},
        )
        self.assertEqual(
            LOOP_ONLY_VARIANTS,
            {"loop_icp_isotropic", "loop_icp_anisotropic", "loop_oriented_anisotropic"},
        )

    def test_proximity_candidates_apply_time_gate_and_endpoint_suppression(self):
        timestamps = np.arange(8, dtype=float) * 10.0
        poses = np.repeat(np.eye(4)[None], 8, axis=0)
        poses[:, 0, 3] = [0, 1, 2, 3, 0.1, 1.1, 2.1, 3.1]
        pairs = proximity_pairs(
            timestamps, poses, radius_m=0.25, min_time_gap_s=30.0,
            max_candidates=10, suppression_frames=0,
        )
        self.assertEqual([(a, b) for a, b, _ in pairs], [(0, 4), (1, 5), (2, 6), (3, 7)])
        suppressed = proximity_pairs(
            timestamps, poses, radius_m=0.25, min_time_gap_s=30.0,
            max_candidates=10, suppression_frames=1,
        )
        self.assertEqual(len(suppressed), 2)

    def test_se3_log_and_factor_error_use_left_target_frame_gap(self):
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
        delta[:3, 3] = [0.1, -0.2, 0.05]
        measured = np.eye(4)
        expected = delta @ measured
        error = factor_error_g2o(measured, np.eye(4), expected)
        np.testing.assert_allclose(error, se3_log_g2o(delta), atol=1e-12)

    def test_calibration_scale_uses_only_preregistered_split(self):
        rows = [
            {"dataset": "cal", "valid": True, "observable_rank": 6, "raw_nis": 100.0},
            {"dataset": "held", "valid": True, "observable_rank": 6, "raw_nis": 1e9},
        ]
        scale = calibration_scale(rows, {"cal"})
        self.assertGreater(scale, 0.04)
        self.assertLess(scale, 0.07)

    def test_truth_interpolation_uses_linear_translation_and_slerp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.txt"
            path.write_text(
                "0 0 0 0 0 0 0 1\n"
                "2 2 0 0 0 0 1 0\n",
                encoding="utf-8",
            )
            poses, valid = interpolated_truth(np.asarray([1.0]), path, 1.1)
            self.assertTrue(valid[0])
            np.testing.assert_allclose(poses[0, :3, 3], [1.0, 0.0, 0.0])
            angle = Rotation.from_matrix(poses[0, :3, :3]).magnitude()
            self.assertAlmostEqual(angle, np.pi / 2.0)

    def test_g2o_anisotropic_edge_round_trip(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "xyz", [0.1, -0.2, 0.3]
        ).as_matrix()
        transform[:3, 3] = [1.0, 2.0, -0.5]
        information = np.arange(36, dtype=float).reshape(6, 6)
        information = information + information.T + np.eye(6) * 100
        line = edge_line(4, 5, transform, information)
        parsed = parse_g2o_edge(line)
        self.assertTrue(parsed["sequential"])
        np.testing.assert_allclose(parsed["transform"], transform, atol=1e-12)
        np.testing.assert_allclose(parsed["information"], information, atol=1e-12)

    def test_information_upper_triangle_round_trip(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(6, 6))
        matrix = matrix @ matrix.T + np.eye(6)
        tokens = format_information(matrix).split()
        np.testing.assert_allclose(parse_information(tokens), matrix, atol=1e-12)

    def test_isotropic_ablation_preserves_block_scales_and_removes_cross_terms(self):
        information = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
        information[0, 4] = information[4, 0] = 2.0
        isotropic = isotropize_information(information)
        np.testing.assert_allclose(np.diag(isotropic), [4, 4, 4, 25, 25, 25])
        self.assertEqual(np.count_nonzero(isotropic - np.diag(np.diag(isotropic))), 0)

    def test_loop_csv_uses_target_to_source_and_translation_rotation_sigmas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loops.csv"
            path.write_text(
                "enabled,source_id,target_id,tx,ty,tz,qx,qy,qz,qw,"
                "sigma_tx,sigma_ty,sigma_tz,sigma_roll_deg,sigma_pitch_deg,sigma_yaw_deg\n"
                "1,8,2,1,2,3,0,0,0,1,0.1,0.2,0.4,1,2,4\n",
                encoding="utf-8",
            )
            row = load_loop_rows(path)[0]
            self.assertEqual((row["target_id"], row["source_id"]), (2, 8))
            np.testing.assert_allclose(row["transform"][:3, 3], [1, 2, 3])
            expected = 1.0 / np.square(
                [0.1, 0.2, 0.4, *np.deg2rad([1.0, 2.0, 4.0])]
            )
            np.testing.assert_allclose(np.diag(row["information"]), expected)


if __name__ == "__main__":
    unittest.main()
