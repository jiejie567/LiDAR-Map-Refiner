from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure.scan_context_io import (  # noqa: E402
    ScanContextConfig,
    _gravity_canonical_rotation,
    load_scan_context_gravity,
    load_scan_context_config,
    make_descriptor_with_mask,
    save_scan_context_database,
)
from manual_loop_closure.trajectory_io import TrajectoryData  # noqa: E402


class ScanContextV7Test(unittest.TestCase):
    def test_python_retrieval_scoring_loads_recorded_v7_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            (session / "runtime_params.yaml").write_text(
                "scan_context:\n"
                "  retrieval_height_offset: 0.23\n"
                "  sector_support_exponent: 0.75\n",
                encoding="utf-8",
            )
            config = load_scan_context_config(session)

        self.assertAlmostEqual(config.retrieval_height_offset, 0.23)
        self.assertAlmostEqual(config.sector_support_exponent, 0.75)

    def test_gravity_sidecar_is_explicit_and_timestamp_checked(self) -> None:
        identity = np.eye(4, dtype=np.float64)
        trajectory = TrajectoryData(
            path=Path("synthetic.tum"),
            timestamps=np.asarray([12.5]),
            positions_xyz=np.zeros((1, 3), dtype=np.float64),
            quats_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
            transforms_world_sensor=identity[None, :, :],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            (session / "scan_context_gravity.csv").write_text(
                "index,stamp,up_x,up_y,up_z\n0,12.5,0,0.6,0.8\n",
                encoding="utf-8",
            )
            gravity = load_scan_context_gravity(session, trajectory)
            np.testing.assert_allclose(gravity, [[0.0, 0.6, 0.8]], atol=1e-12)

            (session / "scan_context_gravity.csv").write_text(
                "index,stamp,up_x,up_y,up_z\n0,13.0,0,0,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "timestamp mismatch"):
                load_scan_context_gravity(session, trajectory)

    def test_gravity_rotation_and_canonical_yaw_reconstruct_pose(self) -> None:
        def rotation(axis: str, angle: float) -> np.ndarray:
            c, s = np.cos(angle), np.sin(angle)
            if axis == "x":
                return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            if axis == "y":
                return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        R_map_body = (
            rotation("z", np.deg2rad(31.0))
            @ rotation("y", np.deg2rad(-13.0))
            @ rotation("x", np.deg2rad(17.0))
        )
        up_body = R_map_body.T @ np.array([0.0, 0.0, 1.0])
        R_g = _gravity_canonical_rotation(up_body)
        R_map_descriptor = R_map_body @ R_g.T
        canonical_yaw = np.arctan2(R_map_descriptor[1, 0], R_map_descriptor[0, 0])
        reconstructed = rotation("z", canonical_yaw) @ R_g
        np.testing.assert_allclose(reconstructed, R_map_body, atol=1e-12)

        upside_down = _gravity_canonical_rotation(np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(upside_down, np.diag([1.0, -1.0, -1.0]), atol=1e-12)

    def test_negative_height_keeps_valid_bit(self) -> None:
        config = ScanContextConfig(
            num_rings=3,
            num_sectors=4,
            max_radius=9.0,
            dual_z_layer_enable=True,
            dual_z_split_height=2.5,
            origin_height_from_ground=1.5,
            min_joint_rings=2,
        )
        descriptor, valid = make_descriptor_with_mask(
            np.asarray([[1.0, 0.0, -2.0]], dtype=np.float64),
            config,
        )
        self.assertEqual(float(descriptor[0, 0]), -0.5)
        self.assertTrue(bool(valid[0, 0]))

    def test_writer_uses_little_endian_bitset(self) -> None:
        config = ScanContextConfig(
            num_rings=3,
            num_sectors=4,
            max_radius=9.0,
            dual_z_layer_enable=True,
            dual_z_split_height=2.5,
            origin_height_from_ground=1.5,
            min_joint_rings=2,
        )
        identity = np.eye(4, dtype=np.float64)
        trajectory = TrajectoryData(
            path=Path("synthetic.tum"),
            timestamps=np.asarray([1.0]),
            positions_xyz=np.zeros((1, 3), dtype=np.float64),
            quats_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
            transforms_world_sensor=identity[None, :, :],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "scans.scd"
            save_scan_context_database(
                path=output,
                trajectory=trajectory,
                local_point_sets=[np.asarray([[1.0, 0.0, -2.0]], dtype=np.float64)],
                config=config,
                gravity_up_body=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            )
            contents = output.read_bytes()

        self.assertTrue(contents.startswith(b"FAST_LIO_SCAN_CONTEXT_DB_V7\n"))
        self.assertIn(
            b"PARAMS 3 4 9 1 2.5 1.5 0.40000000000000002 0.59999999999999998 2 1\n",
            contents,
        )
        self.assertIn(b"MASK_BITS 3\n\x01\x00\x00\nEND_ENTRY\n", contents)


if __name__ == "__main__":
    unittest.main()
