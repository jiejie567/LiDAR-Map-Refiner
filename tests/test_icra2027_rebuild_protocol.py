import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ghostloop_eval"
GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(GUI_DIR))

from icra2027_rebuild import (  # noqa: E402
    CODE_INPUTS,
    DatasetSpec,
    TOOL_ROOT,
    prepare_dataset,
    sha256_file,
)
from evaluate_icra2027_rebuild import (  # noqa: E402
    interpolated_truth_in_keyframe_frame,
    resolve_truth_body_to_keyframe,
    target_to_source_truth_error,
)
from manual_loop_closure.python_optimizer.cli import _build_parser  # noqa: E402
from manual_loop_closure.python_optimizer.information_order import (  # noqa: E402
    gtsam_information_to_g2o,
)
from manual_loop_closure.python_optimizer.optimizer import (  # noqa: E402
    _load_constraints_csv,
    _manual_constraint_information,
)


class Icra2027RebuildProtocolTest(unittest.TestCase):
    def test_code_inventory_covers_complete_runtime_package(self):
        relative = {
            path.relative_to(TOOL_ROOT).as_posix()
            for path in CODE_INPUTS
        }
        for required in (
            "gui/manual_loop_closure/optimizer_backend.py",
            "gui/manual_loop_closure/python_optimizer/cli.py",
            "gui/manual_loop_closure/python_optimizer/loop_weighting.py",
            "gui/manual_loop_closure/python_optimizer/oriented_surface_factors.py",
        ):
            self.assertIn(required, relative)

    def test_prepare_dataset_hashes_inputs_and_inventory_without_copying_clouds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            keyframes = source / "key_point_frame"
            keyframes.mkdir(parents=True)
            (keyframes / "0.pcd").write_bytes(b"pcd-zero")
            (keyframes / "1.pcd").write_bytes(b"pcd-one")
            for name, value in {
                "optimized_poses_tum.txt": "poses\n",
                "pose_graph.g2o": "graph\n",
                "runtime_params.yaml": "runtime\n",
            }.items():
                (source / name).write_text(value, encoding="utf-8")
            truth = root / "truth.txt"
            truth.write_text("truth\n", encoding="utf-8")
            destination = root / "run"
            manifest = prepare_dataset(
                "synthetic",
                DatasetSpec(source, "outdoor", truth, "se3"),
                destination,
            )
            self.assertTrue((destination / "key_point_frame").is_symlink())
            self.assertEqual(manifest["keyframes"]["count"], 2)
            self.assertEqual(
                manifest["input_sha256"]["pose_graph.g2o"],
                sha256_file(destination / "pose_graph.g2o"),
            )
            persisted = json.loads((destination / "input_manifest.json").read_text())
            self.assertEqual(persisted["dataset"], "synthetic")
            self.assertEqual(
                persisted["keyframe_export"]["keyframe_frame_role"],
                "lidar_imu",
            )
            self.assertEqual(
                persisted["keyframe_export"]["provenance"],
                "explicit_legacy_dataset_spec",
            )
            with self.assertRaises(FileExistsError):
                prepare_dataset(
                    "synthetic", DatasetSpec(source, "outdoor", truth, "se3"),
                    destination,
                )

    def test_prepare_dataset_freezes_exported_frame_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "key_point_frame").mkdir(parents=True)
            (source / "key_point_frame/0.pcd").write_bytes(b"pcd")
            for name in (
                "optimized_poses_tum.txt", "pose_graph.g2o",
                "runtime_params.yaml",
            ):
                (source / name).write_text("input\n", encoding="utf-8")
            contract = {
                "schema_version": 1,
                "keyframe_frame_role": "base_link",
                "point_frame_equals_pose_frame": True,
                "pose_timestamp": "scan_end",
                "T_keyframe_lidar_imu": np.eye(4).tolist(),
            }
            (source / "keyframe_frame_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            destination = root / "run"
            manifest = prepare_dataset(
                "synthetic",
                DatasetSpec(source, "outdoor", None, None),
                destination,
            )
            self.assertEqual(
                manifest["keyframe_export"]["keyframe_frame_role"],
                "base_link",
            )
            self.assertEqual(
                manifest["keyframe_export"]["provenance"],
                "exported_sidecar",
            )
            self.assertEqual(
                manifest["keyframe_export"]["sha256"],
                manifest["input_sha256"]["keyframe_frame_contract.json"],
            )

    def test_optimizer_cli_exposes_robust_and_chordal_benchmarks(self):
        parser = _build_parser()
        common = [
            "--session-root", "/tmp/session", "--g2o", "/tmp/g.g2o",
            "--tum", "/tmp/t.tum", "--keyframe-dir", "/tmp/kf",
            "--constraints-csv", "/tmp/c.csv", "--output-dir", "/tmp/out",
        ]
        self.assertEqual(
            parser.parse_args([*common, "--optimize-mode", "gnc_tls"]).optimize_mode,
            "gnc_tls",
        )
        self.assertEqual(
            parser.parse_args([*common, "--optimize-mode", "lm_huber"]).optimize_mode,
            "lm_huber",
        )
        self.assertEqual(
            parser.parse_args(
                [*common, "--optimize-mode", "lm_chordal"]
            ).optimize_mode,
            "lm_chordal",
        )

    def test_constraint_csv_accepts_full_anisotropic_information(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constraints.csv"
            information = np.diag([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
            information[0, 4] = information[4, 0] = 0.25
            upper = [
                float(information[row, col])
                for row in range(6) for col in range(row, 6)
            ]
            fields = [
                "enabled", "source_id", "target_id", "tx", "ty", "tz",
                "qx", "qy", "qz", "qw", "sigma_tx", "sigma_ty",
                "sigma_tz", "sigma_roll_deg", "sigma_pitch_deg",
                "sigma_yaw_deg", "information_upper_json",
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    **{field: "1" for field in fields},
                    "source_id": "4", "target_id": "1",
                    "qx": "0", "qy": "0", "qz": "0", "qw": "1",
                    "information_upper_json": json.dumps(upper),
                })
            constraint = _load_constraints_csv(path)[0]
            self.assertIsNotNone(constraint.information_g2o)
            recovered = gtsam_information_to_g2o(
                _manual_constraint_information(constraint)
            )
            np.testing.assert_allclose(recovered, information)

    def test_truth_error_uses_target_to_source_convention(self):
        world_source = np.eye(4)
        world_source[:3, 3] = [3.0, -1.0, 0.5]
        angle = np.deg2rad(30.0)
        world_source[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        world_target = np.eye(4)
        world_target[:3, 3] = [1.0, 2.0, -0.5]
        measured = np.linalg.inv(world_target) @ world_source
        translation, rotation = target_to_source_truth_error(
            measured, world_source, world_target
        )
        self.assertLess(translation, 1e-12)
        self.assertLess(rotation, 1e-8)
        reversed_measurement = np.linalg.inv(measured)
        wrong_translation, wrong_rotation = target_to_source_truth_error(
            reversed_measurement, world_source, world_target
        )
        self.assertGreater(wrong_translation, 1.0)
        self.assertGreater(wrong_rotation, 5.0)

    def test_truth_is_interpolated_and_moved_to_keyframe_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.tum"
            # World translation 0 -> 2 m and yaw 0 -> 90 deg over two seconds.
            truth.write_text(
                "0 0 0 0 0 0 0 1\n"
                "2 2 0 0 0 0 0.7071067811865475 0.7071067811865476\n",
                encoding="utf-8",
            )
            body_to_keyframe = np.eye(4)
            body_to_keyframe[0, 3] = 1.0
            frame_config = {
                "datasets": {
                    "synthetic": {
                        "T_truth_body_keyframe": body_to_keyframe.tolist()
                    }
                }
            }
            poses, valid, orientation_valid = (
                interpolated_truth_in_keyframe_frame(
                    np.asarray([1.0]), truth, "synthetic", frame_config,
                    max_dt=1.0,
                )
            )
            self.assertTrue(valid[0])
            self.assertTrue(orientation_valid[0])
            expected = np.asarray([
                1.0 + np.sqrt(0.5), np.sqrt(0.5), 0.0
            ])
            np.testing.assert_allclose(poses[0, :3, 3], expected, atol=1e-9)
            yaw = np.arctan2(poses[0, 1, 0], poses[0, 0, 0])
            self.assertAlmostEqual(np.degrees(yaw), 45.0, places=8)

    def test_schema_v2_selects_transform_from_frozen_frame_role(self):
        imu_transform = np.eye(4)
        imu_transform[0, 3] = 0.25
        frame_config = {
            "datasets": {
                "synthetic": {
                    "legacy_keyframe_frame_role": "lidar_imu",
                    "keyframe_frames": {
                        "lidar_imu": {
                            "T_truth_body_keyframe": imu_transform.tolist()
                        },
                        "base_link": {
                            "T_truth_body_keyframe": np.eye(4).tolist()
                        },
                    },
                }
            }
        }
        resolved_imu, imu_metadata = resolve_truth_body_to_keyframe(
            "synthetic", frame_config, "lidar_imu"
        )
        resolved_body, body_metadata = resolve_truth_body_to_keyframe(
            "synthetic", frame_config, "base_link"
        )
        self.assertAlmostEqual(resolved_imu[0, 3], 0.25)
        self.assertAlmostEqual(resolved_body[0, 3], 0.0)
        self.assertEqual(imu_metadata["keyframe_frame_role"], "lidar_imu")
        self.assertEqual(body_metadata["keyframe_frame_role"], "base_link")


if __name__ == "__main__":
    unittest.main()
