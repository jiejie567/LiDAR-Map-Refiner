from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TOOL_DIR = Path(__file__).resolve().parents[1]
GUI_DIR = TOOL_DIR / "gui"
for path in (TOOL_DIR, GUI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manual_loop_closure.python_optimizer.exporters import (  # noqa: E402
    build_map_and_trajectory_from_tum,
    load_xyzi_points,
    save_xyzi_pcd,
)


class CompletePgoPoseExportTest(unittest.TestCase):
    def test_map_trajectory_and_scd_preserve_complete_pgo_pose(self) -> None:
        pgo_rotation = Rotation.from_euler(
            "xyz",
            np.deg2rad([12.0, -7.0, 31.0]),
        )
        pgo_translation = np.asarray([4.0, -2.0, 3.5], dtype=np.float64)
        quat = pgo_rotation.as_quat()
        local_points = np.asarray(
            [
                [1.0, 0.0, -0.4, 10.0],
                [0.0, 2.0, 0.8, 20.0],
                [-1.0, 0.5, 1.2, 30.0],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session"
            keyframes = session / "keyframes"
            output_dir = Path(temp_dir) / "output"
            keyframes.mkdir(parents=True)
            output_dir.mkdir()

            save_xyzi_pcd(keyframes / "0.pcd", local_points)
            tum_path = session / "optimized_poses_tum.txt"
            tum_path.write_text(
                "1.000000000 "
                f"{pgo_translation[0]:.9f} {pgo_translation[1]:.9f} "
                f"{pgo_translation[2]:.9f} "
                f"{quat[0]:.12f} {quat[1]:.12f} {quat[2]:.12f} {quat[3]:.12f}\n",
                encoding="utf-8",
            )
            (session / "runtime_params.yaml").write_text(
                "scan_context:\n"
                "  num_rings: 3\n"
                "  num_sectors: 8\n"
                "  max_radius: 20.0\n"
                "  gravity_canonicalization_enable: true\n",
                encoding="utf-8",
            )
            # Deliberately disagree with the tilted PGO pose. This vector must
            # canonicalize only the descriptor, never overwrite PGO roll/pitch.
            (session / "scan_context_gravity.csv").write_text(
                "index,stamp,up_x,up_y,up_z\n"
                "0,1.0,0,0,1\n",
                encoding="utf-8",
            )

            output_map = output_dir / "scans.pcd"
            output_trajectory = output_dir / "trajectory.pcd"
            output_scd = output_dir / "scans.scd"
            build_map_and_trajectory_from_tum(
                tum_path=tum_path,
                keyframe_dir=keyframes,
                output_map=output_map,
                output_trajectory=output_trajectory,
                voxel_leaf=0.0,
                output_scan_context=output_scd,
            )

            actual_map = load_xyzi_points(output_map)
            expected_xyz = (
                local_points[:, :3].astype(np.float64) @ pgo_rotation.as_matrix().T
                + pgo_translation
            )
            np.testing.assert_allclose(actual_map[:, :3], expected_xyz, atol=1e-6)

            trajectory_points = load_xyzi_points(output_trajectory)
            np.testing.assert_allclose(
                trajectory_points[0, :3],
                pgo_translation,
                atol=1e-6,
            )

            entry_line = next(
                line.decode("utf-8")
                for line in output_scd.read_bytes().splitlines()
                if line.startswith(b"ENTRY ")
            )
            entry = entry_line.split()
            self.assertEqual(entry[0:2], ["ENTRY", "0"])
            np.testing.assert_allclose(
                [float(value) for value in entry[3:6]],
                pgo_translation,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                [float(value) for value in entry[6:9]],
                pgo_rotation.as_euler("xyz"),
                atol=1e-9,
            )

            self.assertFalse((output_dir / "ground_alignment.json").exists())
            self.assertFalse((output_dir / "ground_fit_overview.png").exists())
            self.assertFalse((output_dir / "ground_aligned_poses_tum.txt").exists())


if __name__ == "__main__":
    unittest.main()
