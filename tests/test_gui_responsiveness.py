from __future__ import annotations

import csv
import os
import inspect
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MS_MANUAL_LOOP_REEXEC", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

import manual_loop_closure_tool as tool  # noqa: E402
import manual_loop_closure.open3d_viewer as viewer  # noqa: E402
import manual_loop_closure.optimizer_backend as optimizer_backend  # noqa: E402
import manual_loop_closure.registration as registration  # noqa: E402
from manual_loop_closure.open3d_viewer import (  # noqa: E402
    EmbeddedOpen3DWidget,
    PreviewScene,
)


class GuiResponsivenessTest(unittest.TestCase):
    def test_busy_gui_marker_is_pid_owned_and_clears_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {"GHOSTLOOP_GUI_COMPUTE_DIR": directory}):
            window = tool.ManualLoopClosureWindow()
            window._background_label = "Run GICP"
            window._repair_map_stage = None
            window._sync_gui_compute_marker(True)
            marker = Path(directory) / f"{os.getpid()}.active"
            self.assertEqual(
                marker.read_text(encoding="utf-8"),
                f"{os.getpid()} Run GICP\n")
            window._sync_gui_compute_marker(False)
            self.assertFalse(marker.exists())
            window.cloud_view.shutdown()
            window.deleteLater()

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = EmbeddedOpen3DWidget()
        self.scene = PreviewScene(
            target_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            initial_source_points=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            adjusted_source_points=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
            final_source_points=np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64),
            transform_world_source_initial=np.eye(4),
            transform_world_source_adjusted=np.eye(4),
            transform_world_source_final=np.eye(4),
        )

    def test_async_auto_seed_does_not_fill_raw_frame_cache(self) -> None:
        source = inspect.getsource(
            tool.ManualLoopClosureWindow.run_auto_seed_async)
        self.assertIn("load_local_points_uncached", source)
        self.assertIn("build_descriptor_stack", source)
        self.assertNotIn("build_descriptor_and_prepared_stack", source)
        self.assertNotIn("detect_ghost_regions", source)
        self.assertNotIn("_seed_is_warranted", source)

    def tearDown(self) -> None:
        self.widget.shutdown()
        self.widget.deleteLater()

    def _geometry(self, mode: str, scene: PreviewScene | None = None) -> dict[str, dict]:
        self.widget._display_mode = mode
        items, _ = self.widget._visible_geometry(scene or self.scene)
        return {item["name"]: item for item in items}

    def test_before_and_after_gicp_are_not_reversed(self) -> None:
        before = self._geometry("preview")
        after = self._geometry("final")

        np.testing.assert_array_equal(
            before["source_preview"]["points"], self.scene.adjusted_source_points)
        np.testing.assert_array_equal(
            after["source_final"]["points"], self.scene.final_source_points)
        self.assertNotIn("source_final", before)
        self.assertNotIn("source_preview", after)

    def test_after_gicp_does_not_silently_fall_back_to_before(self) -> None:
        scene_without_result = PreviewScene(
            target_points=self.scene.target_points,
            adjusted_source_points=self.scene.adjusted_source_points,
            final_source_points=None,
        )
        after = self._geometry("final", scene_without_result)

        self.assertNotIn("source_final", after)
        self.assertNotIn("source_preview", after)

    def test_compare_contains_both_before_and_after(self) -> None:
        compare = self._geometry("compare")

        np.testing.assert_array_equal(
            compare["source_compare_preview"]["points"],
            self.scene.adjusted_source_points,
        )
        np.testing.assert_array_equal(
            compare["source_compare_final"]["points"],
            self.scene.final_source_points,
        )

    def test_large_preview_is_capped_only_in_render_scene(self) -> None:
        full_target = np.arange(
            (tool.PREVIEW_TARGET_DISPLAY_CAP_THRESHOLD + 123) * 3,
            dtype=np.float64,
        ).reshape(-1, 3)
        preview = SimpleNamespace(
            target_points_world=full_target,
            target_frame_indices=(),
            source_points_local=np.zeros((1, 3)),
            transform_world_target=np.eye(4),
        )
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window.trajectory = None

        scene = window._make_preview_scene(preview=preview)

        self.assertLessEqual(
            scene.target_points.shape[0], tool.PREVIEW_TARGET_DISPLAY_LIMIT)
        self.assertEqual(preview.target_points_world.shape, full_target.shape)
        self.assertTrue(np.shares_memory(scene.target_points, full_target))

    def test_tensor_voxel_downsample_preserves_legacy_voxel_semantics(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.normal(size=(500_001, 3)) * np.array([20.0, 5.0, 2.0])
        voxel = 0.2
        legacy = registration.numpy_to_open3d(points).voxel_down_sample(voxel)
        legacy_points = np.asarray(legacy.points, dtype=np.float64)

        tensor_points = registration.voxel_downsample(points, voxel)

        self.assertEqual(tensor_points.shape, legacy_points.shape)
        from scipy.spatial import cKDTree
        nearest = cKDTree(legacy_points).query(tensor_points, workers=1)[0]
        self.assertLess(float(np.max(nearest)), 5e-5)

    def test_compact_contiguous_target_keeps_renderer_fast_path(self) -> None:
        compact_target = np.zeros(
            (tool.PREVIEW_TARGET_DISPLAY_CAP_THRESHOLD, 3), dtype=np.float64)
        preview = SimpleNamespace(
            target_points_world=compact_target,
            target_frame_indices=(),
            source_points_local=np.zeros((1, 3)),
            transform_world_target=np.eye(4),
        )
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window.trajectory = None

        scene = window._make_preview_scene(preview=preview)

        self.assertIs(scene.target_points, compact_target)

    def test_preview_summary_matches_compact_target_render_count(self) -> None:
        compact_target = np.zeros(
            (tool.PREVIEW_TARGET_DISPLAY_CAP_THRESHOLD, 3), dtype=np.float64)
        preview = SimpleNamespace(
            target_cloud_mode=tool.TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP,
            target_neighbors=40,
            time_gap_filter_enabled=False,
            time_gap_filter_applied=False,
            target_frame_range=(1, 40),
            target_window_clipped=False,
            target_frame_count=40,
            target_point_count=compact_target.shape[0],
            target_points_world=compact_target,
        )
        window = tool.ManualLoopClosureWindow()

        window._update_preview_summary(preview)

        self.assertNotIn("view", window.target_map_label.text())
        self.assertIn(
            f"display_points={compact_target.shape[0]}",
            window.target_map_label.toolTip(),
        )
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_environment_preset_scales_target_preview_work(self) -> None:
        window = tool.ManualLoopClosureWindow()
        window.env_combo.setCurrentIndex(
            window.env_combo.findData("outdoor"))

        self.assertEqual(window.target_neighbors_spin.value(), 40)
        self.assertAlmostEqual(window.target_map_voxel_spin.value(), 0.40)
        self.assertAlmostEqual(window.voxel_spin.value(), 0.40)
        self.assertEqual(window.balm_iter_spin.value(), 20)
        self.assertAlmostEqual(window._balm_downsample_leaf, 0.40)
        self.assertAlmostEqual(window._balm_max_range, 80.0)
        balm = window._balm_default_params()
        self.assertAlmostEqual(balm.root_voxel_size, 4.0)
        self.assertAlmostEqual(balm.downsample_leaf, 0.40)
        self.assertAlmostEqual(balm.max_range, 80.0)
        self.assertTrue(window.balm_double_sided_check.isChecked())
        self.assertTrue(balm.double_sided_enable)
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_fresh_gui_requires_environment_but_keeps_safe_indoor_widget_values(self) -> None:
        window = tool.ManualLoopClosureWindow()

        self.assertIsNone(window.env_combo.currentData())
        self.assertAlmostEqual(window.voxel_spin.value(), 0.20)
        self.assertAlmostEqual(window.target_map_voxel_spin.value(), 0.20)
        self.assertAlmostEqual(window.balm_voxel_spin.value(), 1.0)
        self.assertEqual(window.balm_iter_spin.value(), 20)
        self.assertAlmostEqual(window._balm_downsample_leaf, 0.20)
        self.assertAlmostEqual(window._balm_max_range, 80.0)
        self.assertTrue(window.balm_double_sided_check.isChecked())
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_constraint_csv_converts_rotation_variance_rad2_to_degrees(self) -> None:
        constraint = tool.ManualConstraint(
            manual_uid=1,
            enabled=True,
            source_id=1,
            target_id=0,
            target_cloud_mode=tool.TARGET_CLOUD_MODE_TEMPORAL_WINDOW,
            target_neighbors=1,
            min_time_gap_sec=0.0,
            target_map_voxel_size=0.2,
            transform_world_source_final=np.eye(4),
            transform_target_source_final=np.eye(4),
            source_points_world_final=np.zeros((0, 3)),
            fitness=1.0,
            inlier_rmse=0.0,
            variance_t_m2=(1.6, 1.6, 1.6),
            variance_r_rad2=(0.1, 0.1, 0.1),
        )

        row = constraint.csv_row()

        self.assertAlmostEqual(float(row[13]), np.degrees(np.sqrt(0.1)), places=5)
        self.assertAlmostEqual(float(row[14]), np.degrees(np.sqrt(0.1)), places=5)
        self.assertAlmostEqual(float(row[15]), np.degrees(np.sqrt(0.1)), places=5)

    def test_gravity_sanity_error_is_yaw_invariant(self) -> None:
        yaw = np.deg2rad(90.0)
        transform = np.eye(4)
        transform[:3, :3] = [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.assertAlmostEqual(
            tool._gravity_error_deg(
                transform, np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
            0.0,
        )

    def test_automatic_registration_result_drops_render_only_arrays(self) -> None:
        points = np.ones((10, 3), dtype=np.float64)
        preview = SimpleNamespace(
            target_points_world=points,
            source_points_local=points,
            source_points_world_initial=points,
            source_points_world_adjusted=points,
        )
        # dataclasses.replace is used in production; a real preview/result is
        # the smallest honest fixture for that contract.
        preview = registration.RegistrationPreview(
            source_id=1, target_id=0,
            target_cloud_mode=registration.TARGET_CLOUD_MODE_TEMPORAL_WINDOW,
            target_neighbors=1, min_time_gap_sec=0.0,
            target_map_voxel_size=0.2, target_frame_indices=(0,),
            time_gap_filter_enabled=False, time_gap_filter_applied=False,
            target_points_world=points, source_points_local=points,
            source_points_world_initial=points,
            source_points_world_adjusted=points,
            transform_world_source_initial=np.eye(4),
            transform_world_source_adjusted=np.eye(4),
            transform_world_target=np.eye(4),
        )
        result = registration.RegistrationResult(
            preview=preview,
            transform_world_source_final=np.eye(4),
            transform_target_source_final=np.eye(4),
            source_points_world_final=points,
            fitness=1.0, inlier_rmse=0.1,
        )

        compact = tool._compact_registration_result(result)

        self.assertEqual(compact.preview.target_points_world.shape, (0, 3))
        self.assertEqual(compact.preview.source_points_local.shape, (0, 3))
        self.assertEqual(compact.source_points_world_final.shape, (0, 3))
        np.testing.assert_array_equal(
            compact.transform_target_source_final,
            result.transform_target_source_final,
        )

    def test_target_cache_is_bounded_lru(self) -> None:
        self.assertLessEqual(registration.TARGET_CACHE_MAX_ENTRIES, 2)

    def test_balm_adoption_fails_closed_when_output_cannot_be_validated(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window._balm_source_run_dir = root / "pgo"
            window._last_output_dir = root / "balm"
            window.append_log = mock.Mock()

            self.assertFalse(window._balm_output_is_sane())
            self.assertTrue(window.append_log.called)

    def test_balm_adoption_rejects_in_place_rotation_degeneracy(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pgo = root / "pgo"
            balm = root / "balm"
            pgo.mkdir()
            balm.mkdir()
            (pgo / "optimized_poses_tum.txt").write_text(
                "0 0 0 0 0 0 0 1\n", encoding="utf-8"
            )
            # Twenty degrees about z with no translation: a translation-only
            # adoption gate would silently accept this planar degeneracy.
            angle = np.deg2rad(20.0) / 2.0
            (balm / "optimized_poses_tum.txt").write_text(
                f"0 0 0 0 0 0 {np.sin(angle)} {np.cos(angle)}\n",
                encoding="utf-8",
            )
            window._balm_source_run_dir = pgo
            window._last_output_dir = balm
            window.append_log = mock.Mock()

            self.assertFalse(window._balm_output_is_sane())
            self.assertAlmostEqual(
                window._last_balm_motion_stats[
                    "measured_mean_pose_rotation_deg"
                ],
                20.0,
                places=6,
            )

    def test_optimizer_snapshots_last_valid_run_before_new_output(self) -> None:
        source = inspect.getsource(
            tool.ManualLoopClosureWindow.run_optimization
        )
        snapshot = source.index(
            "self._pre_optimize_snapshot = self._capture_undo_snapshot()"
        )
        redirect = source.index("self._last_output_dir = output_dir")
        self.assertLess(snapshot, redirect)

    def test_balm_backend_receives_environment_density_and_range(self) -> None:
        options = tool.BalmRunOptions(
            tum_path=Path("poses.tum"),
            keyframe_dir=Path("keyframes"),
            output_dir=Path("output"),
            downsample_leaf=0.5,
            max_range=60.0,
        )

        args = options.to_cli_args()

        self.assertEqual(args[args.index("--downsample-leaf") + 1], "0.500000")
        self.assertEqual(args[args.index("--max-range") + 1], "60.000000")
        self.assertEqual(args[args.index("--max-iterations") + 1], "20")
        self.assertIn("--double-sided", args)
        self.assertNotIn("--diagnose-map-inconsistency", args)
        self.assertIn(
            "--diagnose-map-inconsistency",
            tool.BalmRunOptions(
                tum_path=Path("poses.tum"),
                keyframe_dir=Path("keyframes"),
                output_dir=Path("output"),
                diagnose_map_inconsistency=True,
            ).to_cli_args(),
        )
        self.assertIn(
            "--no-double-sided",
            tool.BalmRunOptions(
                tum_path=Path("poses.tum"),
                keyframe_dir=Path("keyframes"),
                output_dir=Path("output"),
                double_sided_enable=False,
            ).to_cli_args(),
        )

    def test_map_inconsistency_diagnostics_are_off_by_default(self) -> None:
        window = tool.ManualLoopClosureWindow()
        self.assertEqual(window.auto_seed_button.text(), "Initial Loop Search")
        self.assertEqual(window.optimize_button.text(), "Audited PGO")
        self.assertEqual(window.balm_button.text(), "Final Refinement")
        self.assertEqual(
            window.map_diagnostics_check.text(), "Duplicated-surface audit"
        )
        self.assertFalse(window.map_diagnostics_check.isChecked())
        self.assertIn(
            "inspection only",
            window.map_diagnostics_check.toolTip().lower(),
        )
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_balm_summary_does_not_call_plateau_pose_convergence(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "balm_report.json").write_text(
                json.dumps({
                    "termination": {
                        "reason": "final_stage_objective_plateau",
                        "completed_updates": 6,
                        "requested_updates": 20,
                    },
                    "stages": [{
                        "iterations": [{
                            "max_translation_update_m": 0.0092,
                            "max_rotation_update_deg": 0.04,
                        }],
                    }],
                    "params": {
                        "map_inconsistency_diagnostics_enabled": False,
                    },
                }),
                encoding="utf-8",
            )
            window._last_output_dir = run_dir
            summary = window._balm_report_summary()

        self.assertIn("not certified as pose convergence", summary)
        self.assertIn("duplicated-surface audit disabled", summary)
        self.assertIn("0.92 cm", summary)

    def test_constant_rgb_export_preserves_xyz_and_replaces_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pcd"
            destination = Path(directory) / "colored.pcd"
            header = (
                "# .PCD v0.7\nVERSION 0.7\n"
                "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
                "COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"
            ).encode("ascii")
            points = np.asarray(
                [[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0]],
                dtype="<f4",
            )
            source.write_bytes(header + points.tobytes())

            tool.colorize_binary_xyzi_pcd(
                source, destination, (249, 115, 22)
            )

            payload = destination.read_bytes()
            marker = b"DATA binary\n"
            header_out, binary = payload.split(marker, 1)
            self.assertIn(b"FIELDS x y z rgb", header_out)
            words = np.frombuffer(binary, dtype="<u4").reshape(-1, 4)
            np.testing.assert_array_equal(words[:, :3], points.view("<u4")[:, :3])
            self.assertTrue(np.all(words[:, 3] == 0xF97316))

    def test_gui_python_candidate_preserves_venv_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "venv/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.symlink_to(Path(sys.executable))
            with mock.patch.dict(
                    os.environ,
                    {"MANUAL_LOOP_GUI_PYTHON": str(candidate)}):
                candidates = tool._candidate_python_executables()

            self.assertIn(candidate.absolute(), candidates)
            self.assertNotEqual(candidate.absolute(), candidate.resolve())

    def test_gui_dependency_probe_fails_closed_after_timeout(self) -> None:
        with mock.patch.object(
            tool.subprocess,
            "run",
            side_effect=tool.subprocess.TimeoutExpired("python", 10.0),
        ):
            supported = tool._python_supports_manual_loop_dependencies(
                Path("/stalled/venv/bin/python")
            )

        self.assertFalse(supported)

    def test_optimizer_dependency_probe_fails_closed_after_timeout(self) -> None:
        with mock.patch.object(
            optimizer_backend.subprocess,
            "run",
            side_effect=optimizer_backend.subprocess.TimeoutExpired("python", 10.0),
        ):
            supported = optimizer_backend._supports_imports(
                Path("/stalled/venv/bin/python"), "import numpy"
            )

        self.assertFalse(supported)

    def test_manual_drag_replaces_only_source_geometry(self) -> None:
        class FakeScene:
            def __init__(self) -> None:
                self.geometries = {
                    "target": "large-target",
                    "source_preview": "old-source",
                    "axis_source": "old-axis",
                }
                self.removed: list[str] = []

            def has_geometry(self, name: str) -> bool:
                return name in self.geometries

            def remove_geometry(self, name: str) -> None:
                self.removed.append(name)
                self.geometries.pop(name, None)

            def add_geometry(self, name: str, geometry, _material) -> None:
                self.geometries[name] = geometry

            def clear_geometry(self) -> None:
                raise AssertionError("manual drag must retain the target geometry")

        fake_scene = FakeScene()
        self.widget._renderer = SimpleNamespace(scene=fake_scene)
        self.widget._display_mode = "preview"
        self.widget._scene_extent = 10.0
        self.widget._scene = PreviewScene(
            target_points=np.asarray([[0.0, 0.0, 0.0]]),
            editable_source_points_local=np.asarray([[1.0, 2.0, 3.0]]),
            adjusted_source_points=np.asarray([[1.0, 2.0, 3.0]]),
        )
        transform = np.eye(4)
        transform[:3, 3] = (4.0, 5.0, 6.0)

        with (
            mock.patch.object(viewer, "_point_cloud", side_effect=lambda points, _colors=None: points.copy()),
            mock.patch.object(viewer, "_point_material", return_value="point-material"),
            mock.patch.object(viewer, "_axis_mesh", return_value="axis-mesh"),
            mock.patch.object(viewer, "_mesh_material", return_value="mesh-material"),
        ):
            updated = self.widget._replace_editable_source_geometry(transform)

        self.assertTrue(updated)
        self.assertEqual(fake_scene.geometries["target"], "large-target")
        self.assertEqual(fake_scene.removed, ["source_preview", "axis_source"])
        np.testing.assert_array_equal(
            fake_scene.geometries["source_preview"],
            np.asarray([[5.0, 7.0, 9.0]]),
        )
        self.assertEqual(fake_scene.geometries["axis_source"], "axis-mesh")
        np.testing.assert_array_equal(self.widget._active_source_transform, transform)
        # Restore the normal no-renderer test state before tearDown; the fake
        # intentionally rejects full-scene clears to prove this code path did
        # not use one.
        self.widget._renderer = None

    def test_background_task_reports_progress_while_gui_events_continue(self) -> None:
        gui_thread = QtCore.QThread.currentThread()
        worker_thread_seen: list[QtCore.QThread] = []
        progress: list[tuple[str, int, int]] = []
        results: list[str] = []
        gui_ticks: list[int] = []

        def work(report_progress):
            worker_thread_seen.append(QtCore.QThread.currentThread())
            for current in range(1, 4):
                report_progress("GICP", current, 3)
                time.sleep(0.02)
            return "done"

        thread = QtCore.QThread()
        worker = tool.BackgroundTask(work)
        worker.moveToThread(thread)
        event_loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: gui_ticks.append(1))
        worker.progress.connect(
            lambda text, current, total: progress.append((text, current, total)))
        worker.finished.connect(results.append)
        worker.finished.connect(thread.quit)
        worker.finished.connect(event_loop.quit)
        thread.started.connect(worker.run)
        timer.start()
        thread.start()
        QtCore.QTimer.singleShot(2000, event_loop.quit)
        event_loop.exec_()
        timer.stop()
        thread.quit()
        self.assertTrue(thread.wait(1000), "background thread did not stop")

        self.assertEqual(results, ["done"])
        self.assertEqual(progress[-1], ("GICP", 3, 3))
        self.assertIsNot(worker_thread_seen[0], gui_thread)
        self.assertGreater(len(gui_ticks), 0, "GUI event loop did not advance")

    def test_repair_without_seeds_still_creates_optimizer_baseline(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(tool.ManualLoopClosureWindow)
        window._repair_map_stage = "seed"
        window._repair_map_stop_requested = False
        window._balm_ghost_suggestions = []
        window.appended_logs = []
        window.append_log = window.appended_logs.append
        window._set_repair_progress = mock.Mock()

        with mock.patch.object(QtCore.QTimer, "singleShot") as single_shot:
            window._repair_map_after_seed(0)

        self.assertEqual(window._repair_map_stage, "optimize")
        self.assertEqual(window._repair_map_seeds, 0)
        self.assertEqual(single_shot.call_count, 1)
        self.assertIn("audited PGO baseline", " ".join(window.appended_logs))

    def test_repair_button_cancels_active_balm_and_retains_pgo(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._repair_map_stage = "balm"
        window._repair_map_stop_requested = False
        process = mock.Mock()
        window._optimizer_process = process
        window._background_thread = None
        window.repair_map_button = mock.Mock()
        window._set_repair_progress = mock.Mock()
        window.append_log = mock.Mock()
        window._update_session_status_widgets = mock.Mock()

        with mock.patch.object(QtCore.QTimer, "singleShot") as single_shot:
            window.run_repair_map()

        self.assertTrue(window._repair_map_stop_requested)
        self.assertTrue(window._balm_cancel_requested)
        process.terminate.assert_called_once_with()
        single_shot.assert_called_once()
        window.repair_map_button.setText.assert_called_once_with(
            "Stopping Repair…")
        window.repair_map_button.setEnabled.assert_called_once_with(False)
        window._set_repair_progress.assert_called_once_with(
            "Stopping Final Map Refinement · retaining PGO", busy=True)

    def test_canonical_result_contract_rejects_tampered_artifact(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            run = session / "manual_loop_runs" / "terminal"
            run.mkdir(parents=True)
            inputs = {
                "optimized_tum": session / "optimized_poses_tum.txt",
                "pose_graph_g2o": session / "pose_graph.g2o",
            }
            artifacts = {
                "optimized_tum": run / "optimized_poses_tum.txt",
                "pose_graph_g2o": run / "pose_graph.g2o",
                "constraints_csv": run / "manual_loop_constraints.csv",
            }
            for index, path in enumerate([*inputs.values(), *artifacts.values()]):
                path.write_text(f"artifact-{index}\n", encoding="utf-8")
            ledger = session / "proposal_ledgers" / "run.jsonl"
            ledger.parent.mkdir()
            ledger.write_text("{}\n", encoding="utf-8")
            contract = run.parent / ".contract.json"
            payload = {
                "schema": "lidar-map-refiner/headless-result",
                "schema_version": 1,
                "status": "complete",
                "mode": "production",
                "environment": "indoor",
                "session": str(session),
                "terminal_output_dir": str(run),
                "proposal_ledger_jsonl": str(ledger),
                "proposal_ledger_sha256": window._sha256_path(ledger),
                **{key: str(path) for key, path in artifacts.items()},
                "input_sha256": {
                    key: window._sha256_path(path)
                    for key, path in inputs.items()
                },
                "output_sha256": {
                    key: window._sha256_path(path)
                    for key, path in artifacts.items()
                },
            }
            contract.write_text(json.dumps(payload), encoding="utf-8")
            window.session_paths = SimpleNamespace(session_root=session)
            window._canonical_repair_contract = contract
            window._canonical_repair_environment = "indoor"

            self.assertEqual(
                window._validated_canonical_contract()["terminal_output_dir"],
                str(run),
            )
            artifacts["optimized_tum"].write_text(
                "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                window._validated_canonical_contract()

    def test_canonical_repair_stop_kills_isolated_process_group(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        process = mock.Mock()
        process.state.return_value = QtCore.QProcess.Running
        process.processId.return_value = 4242
        window._repair_map_stage = "canonical"
        window._repair_map_stop_requested = False
        window._canonical_repair_process_grouped = True
        window._canonical_repair_cancelled = False
        window._optimizer_process = process
        window.repair_map_button = mock.Mock()
        window._set_repair_progress = mock.Mock()
        window.append_log = mock.Mock()

        with mock.patch.object(os, "killpg") as killpg, mock.patch.object(
            QtCore.QTimer, "singleShot"
        ):
            window.run_repair_map()

        killpg.assert_called_once_with(4242, tool.signal.SIGTERM)
        self.assertTrue(window._canonical_repair_cancelled)
        self.assertTrue(window._repair_map_stop_requested)

    def test_canonical_constraints_restore_csv_measurements_and_units(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window.target_neighbors_spin = mock.Mock()
        window.target_neighbors_spin.value.return_value = 5
        window.target_map_voxel_spin = mock.Mock()
        window.target_map_voxel_spin.value.return_value = 0.2
        window._working_revision = 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tum = root / "poses.txt"
            tum.write_text(
                "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            constraints_csv = root / "constraints.csv"
            with constraints_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(tool.CONSTRAINT_CSV_HEADER)
                writer.writerow([
                    "1", "1", "0", "0.5", "0", "0",
                    "0", "0", "0", "1",
                    "2", "2", "2", "10", "10", "10",
                ])
            ledger = root / "ledger.jsonl"
            ledger.write_text(json.dumps({
                "source_id": 1,
                "target_id": 0,
                "gicp": {"fitness": 0.8, "inlier_rmse_m": 0.12},
            }) + "\n", encoding="utf-8")

            restored = window._canonical_constraints_from_csv(
                constraints_csv, tool.load_tum_trajectory(tum), ledger)

        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0].enabled)
        self.assertAlmostEqual(restored[0].transform_target_source_final[0, 3], 0.5)
        self.assertEqual(restored[0].variance_t_m2, (4.0, 4.0, 4.0))
        self.assertAlmostEqual(
            restored[0].variance_r_rad2[0], np.deg2rad(10.0) ** 2)
        self.assertAlmostEqual(restored[0].fitness, 0.8)
        self.assertAlmostEqual(restored[0].inlier_rmse, 0.12)

    def test_cancelled_balm_retains_pgo_even_if_worker_exits_zero(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow
        )
        pgo = Path("pgo-result")
        window.optimize_button = mock.Mock()
        window.load_button = mock.Mock()
        window._optimizer_heartbeat_timer = mock.Mock()
        window._optimizer_started_at = None
        window._balm_cancel_requested = True
        window._pre_optimize_snapshot = SimpleNamespace(last_output_dir=pgo)
        window._last_output_dir = Path("partial-balm-result")
        window._optimizer_process = mock.Mock()
        window._repair_map_stage = "balm"
        window.append_log = mock.Mock()
        window._update_session_status_widgets = mock.Mock()
        window._repair_map_finish = mock.Mock()
        window._apply_working_optimization_result = mock.Mock()

        window._balm_finished(0, QtCore.QProcess.NormalExit)

        self.assertEqual(window._last_output_dir, pgo)
        window._apply_working_optimization_result.assert_not_called()
        window._repair_map_finish.assert_called_once_with(
            "stopped during Final Map Refinement; retained PGO"
        )

    def test_repair_stop_preserves_completed_stage_and_does_not_advance(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._repair_map_stage = "optimize"
        window._repair_map_stop_requested = True
        window._repair_map_finish = mock.Mock()

        handled = window._repair_map_advance("optimize", True)

        self.assertTrue(handled)
        window._repair_map_finish.assert_called_once_with(
            "stopped by user after audited PGO")

    def test_queued_repair_stage_does_not_start_after_stop(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._repair_map_stage = "balm"
        window._repair_map_stop_requested = True
        window._repair_map_finish = mock.Mock()
        window.run_balm_refinement = mock.Mock()

        window._repair_map_start_balm()

        window.run_balm_refinement.assert_not_called()
        window._repair_map_finish.assert_called_once_with(
            "stopped by user after audited PGO")

    def test_close_does_not_kill_active_repair_writer(self) -> None:
        window = tool.ManualLoopClosureWindow()
        process = mock.Mock()
        event = mock.Mock()
        window._optimizer_process = process
        window._repair_map_stage = "balm"

        with mock.patch.object(QtWidgets.QMessageBox, "information"):
            window.closeEvent(event)

        event.ignore.assert_called_once()
        process.kill.assert_not_called()
        window._optimizer_process = None
        window._repair_map_stage = None
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_close_defaults_to_waiting_for_standalone_optimizer(self) -> None:
        window = tool.ManualLoopClosureWindow()
        process = mock.Mock()
        event = mock.Mock()
        window._optimizer_process = process
        window._repair_map_stage = None

        with mock.patch.object(
                QtWidgets.QMessageBox, "warning",
                return_value=QtWidgets.QMessageBox.No):
            window.closeEvent(event)

        event.ignore.assert_called_once()
        process.kill.assert_not_called()
        window._optimizer_process = None
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_repair_seed_progress_stays_inside_first_stage(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(tool.ManualLoopClosureWindow)
        window._background_thread = object()
        window._background_label = tool.INITIAL_LOOP_SEARCH_LABEL
        window._repair_map_stage = "seed"
        window._set_repair_progress = mock.Mock()
        window.gicp_metrics_label = mock.Mock()

        window._on_background_task_progress(
            "Initial Loop Search · verified 8/8", 8, 8
        )

        window._set_repair_progress.assert_called_once_with(
            "Repair Map · Initial Loop Search · verified 8/8", value=11)

    def test_repair_stops_after_one_balm_even_with_ghost_diagnostics(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._repair_map_stage = "balm"
        window._repair_map_stop_requested = False
        window._last_balm_adopted = True
        window._balm_ghost_suggestions = [{"source_id": 20, "target_id": 2}]
        window.append_log = mock.Mock()
        window._repair_map_finish = mock.Mock()
        window._repair_map_start_auto_loop = mock.Mock()

        handled = window._repair_map_advance("balm", True)

        self.assertTrue(handled)
        window._repair_map_finish.assert_called_once_with(
            "Final Map Refinement adopted"
        )
        window._repair_map_start_auto_loop.assert_not_called()

    def test_repair_balm_failure_is_handled_without_false_success(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._repair_map_stage = "balm"
        window._repair_map_stop_requested = False
        window.append_log = mock.Mock()
        window._repair_map_finish = mock.Mock()

        handled = window._repair_map_advance("balm", False)

        self.assertTrue(handled)
        self.assertFalse(window._repair_map_balm_succeeded)
        window._repair_map_finish.assert_called_once_with(
            "Final Map Refinement failed; retained PGO"
        )

    def test_balm_adoption_fails_closed_without_comparable_trajectories(self) -> None:
        window = tool.ManualLoopClosureWindow.__new__(
            tool.ManualLoopClosureWindow)
        window._prev_ghost_badness = 0.0
        window._balm_source_run_dir = None
        window._last_output_dir = None
        window.append_log = mock.Mock()

        self.assertFalse(window._balm_output_is_sane())
        window.append_log.assert_called_once()

    def test_headless_repair_has_no_unconditional_remeasurement_tail(self) -> None:
        source = (
            REPO_ROOT
            / "scripts/ghostloop_eval/auto_repair_headless.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("MAX_REMEASURE", source)
        self.assertNotIn("for remeasure_pass", source)
        self.assertIn("'new_accepted_loop_only'", source)
        self.assertIn("'single_final_double_sided_balm'", source)

    def test_clean_graph_is_allowed_only_for_repair_baseline(self) -> None:
        window = tool.ManualLoopClosureWindow()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyframes = root / "keyframes"
            keyframes.mkdir()
            tum = root / "optimized_poses_tum.txt"
            tum.write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
            graph_path = root / "pose_graph.g2o"
            raw_line = "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
            graph_path.write_text(raw_line, encoding="utf-8")
            graph = tool.PoseGraphData(
                path=graph_path,
                vertex_ids=[0],
                positions_xyz=np.zeros((1, 3)),
                edge_records=[], odom_edges=[], loop_edges=[],
                prior_nodes=set(), gnss_nodes={}, raw_lines=[raw_line],
            )
            window.session_paths = SimpleNamespace(
                session_root=root, keyframe_dir=keyframes)
            window.pose_graph = graph
            window.original_pose_graph = graph
            window.original_trajectory = SimpleNamespace(path=tum)
            window.constraints = []
            window.disabled_loop_changes = {}
            window._session_dirty = False
            window._repair_map_stage = "optimize"
            window._resolve_optimizer_backend = mock.Mock(
                return_value=SimpleNamespace(display_name="test"))
            window._start_optimizer_backend = mock.Mock()

            window.run_optimization()

            window._start_optimizer_backend.assert_called_once()
            self.assertIsNotNone(window._last_output_dir)
            self.assertTrue(
                (window._last_output_dir / "manual_loop_constraints.csv").is_file())

        window.cloud_view.shutdown()
        window.deleteLater()

    def test_window_background_task_lifecycle_applies_result_without_waiting(self) -> None:
        window = tool.ManualLoopClosureWindow()
        applied: list[str] = []
        event_loop = QtCore.QEventLoop()

        def work(report_progress):
            report_progress("GICP", 1, 1)
            return "result"

        def apply(value):
            applied.append(value)
            event_loop.quit()

        self.assertTrue(window._start_background_task("Run GICP", work, apply))
        QtCore.QTimer.singleShot(3000, event_loop.quit)
        event_loop.exec_()

        self.assertEqual(applied, ["result"])
        self.assertIsNone(window._background_thread)
        self.assertIsNone(window._background_worker)
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_background_apply_failure_resets_repair_workflow(self) -> None:
        window = tool.ManualLoopClosureWindow()
        window._background_label = tool.INITIAL_LOOP_SEARCH_LABEL
        window._background_result = object()
        window._background_error = None
        window._background_on_finished = mock.Mock(
            side_effect=RuntimeError("apply failed"))
        window._background_thread = None
        window._background_worker = None
        window._repair_map_stage = "seed"
        window._repair_map_finish = mock.Mock()

        with mock.patch.object(window, "_show_error"):
            window._finalize_background_task()

        window._repair_map_finish.assert_called_once_with(
            "Initial Loop Search apply failed")
        self.assertIn(
            "apply failed", window.log_text.toPlainText().lower())
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_point_cloud_preview_build_runs_off_gui_thread(self) -> None:
        window = tool.ManualLoopClosureWindow()
        gui_thread = QtCore.QThread.currentThread()
        worker_threads: list[QtCore.QThread] = []
        gui_ticks: list[int] = []
        applied: list[str] = []

        class SlowWorkspace:
            def build_preview(self, **_kwargs):
                worker_threads.append(QtCore.QThread.currentThread())
                time.sleep(0.08)
                return "preview"

        window.workspace = SlowWorkspace()
        event_loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: gui_ticks.append(1))

        def apply(value):
            applied.append(value)
            event_loop.quit()

        timer.start()
        self.assertTrue(window._start_preview_build(
            source_id=20,
            target_id=10,
            delta_transform=np.eye(4),
            target_cloud_mode="temporal_window",
            target_neighbors=100,
            min_time_gap_sec=30.0,
            target_map_voxel_size=0.1,
            on_finished=apply,
        ))
        self.assertFalse(window.load_button.isEnabled())
        QtCore.QTimer.singleShot(3000, event_loop.quit)
        event_loop.exec_()
        timer.stop()

        self.assertEqual(applied, ["preview"])
        self.assertIsNot(worker_threads[0], gui_thread)
        self.assertGreater(len(gui_ticks), 2)
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_session_file_validation_runs_off_gui_thread(self) -> None:
        window = tool.ManualLoopClosureWindow()
        window.session_root_edit.clear()
        window.g2o_edit.clear()
        gui_thread = QtCore.QThread.currentThread()
        worker_threads: list[QtCore.QThread] = []
        gui_ticks: list[int] = []
        event_loop = QtCore.QEventLoop()
        fake_paths = SimpleNamespace(
            session_root=Path("/tmp/session"),
            g2o_path=Path("/tmp/session/pose_graph.g2o"),
            tum_path=Path("/tmp/session/poses.txt"),
            keyframe_dir=Path("/tmp/session/keyframes"),
        )
        fake_graph = SimpleNamespace(vertex_ids=[0])
        fake_trajectory = SimpleNamespace(size=1)

        def slow_graph(_path):
            worker_threads.append(QtCore.QThread.currentThread())
            time.sleep(0.08)
            return fake_graph

        def install_payload():
            event_loop.quit()

        window._load_session_preloaded = install_payload
        timer = QtCore.QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: gui_ticks.append(1))
        with (
            mock.patch.object(tool, "resolve_session_paths", return_value=fake_paths),
            mock.patch.object(tool, "load_pose_graph", side_effect=slow_graph),
            mock.patch.object(tool, "load_tum_trajectory", return_value=fake_trajectory),
            mock.patch.object(tool, "list_numbered_pcds", return_value=[Path("0.pcd")]),
            mock.patch.object(tool, "align_pose_graph_to_frame_count", return_value=(fake_graph, None)),
            mock.patch.object(tool, "validate_keyframe_numbering", return_value=None),
        ):
            timer.start()
            window.load_session()
            self.assertFalse(window.load_button.isEnabled())
            QtCore.QTimer.singleShot(3000, event_loop.quit)
            event_loop.exec_()
            timer.stop()

        self.assertIsNot(worker_threads[0], gui_thread)
        self.assertGreater(len(gui_ticks), 2)
        # The completion callback has installed the validated payload and the
        # common finalizer restores the load button.
        self.assertIsNone(window._background_thread)
        self.assertTrue(window.load_button.isEnabled())
        self.assertEqual(window._preloaded_session_payload[0], fake_paths)
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_preview_edit_during_build_queues_one_latest_refresh(self) -> None:
        window = tool.ManualLoopClosureWindow()
        window.workspace = SimpleNamespace(
            build_preview=lambda **_kwargs: (time.sleep(0.05), "preview")[1])
        window.source_id = 20
        window.target_id = 10
        event_loop = QtCore.QEventLoop()

        self.assertTrue(window._start_preview_build(
            source_id=20,
            target_id=10,
            delta_transform=np.eye(4),
            target_cloud_mode="temporal_window",
            target_neighbors=100,
            min_time_gap_sec=30.0,
            target_map_voxel_size=0.1,
            on_finished=lambda _value: event_loop.quit(),
        ))
        window.schedule_preview_refresh()
        window.schedule_preview_refresh(reset_camera=True)
        self.assertTrue(window._preview_refresh_queued)

        QtCore.QTimer.singleShot(3000, event_loop.quit)
        event_loop.exec_()

        self.assertFalse(window._preview_refresh_queued)
        self.assertTrue(window._preview_timer.isActive())
        self.assertTrue(window._pending_preview_reset_camera)
        window._preview_timer.stop()
        window.cloud_view.shutdown()
        window.deleteLater()

    def test_export_map_build_runs_off_gui_thread(self) -> None:
        window = tool.ManualLoopClosureWindow()
        gui_thread = QtCore.QThread.currentThread()
        worker_threads: list[QtCore.QThread] = []
        gui_ticks: list[int] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            keyframe_dir = root / "keyframes"
            run_dir.mkdir()
            keyframe_dir.mkdir()
            window.session_paths = SimpleNamespace(
                session_root=root, keyframe_dir=keyframe_dir)
            window._last_output_dir = run_dir
            window._session_dirty = False

            def slow_export(*args, **kwargs):
                worker_threads.append(QtCore.QThread.currentThread())
                time.sleep(0.08)

            window._ensure_run_map_outputs = slow_export
            event_loop = QtCore.QEventLoop()
            timer = QtCore.QTimer()
            timer.setInterval(5)
            timer.timeout.connect(lambda: gui_ticks.append(1))

            with mock.patch.object(
                    QtWidgets.QMessageBox, "information", return_value=None):
                timer.start()
                window.export_final_result()

                def finish_when_exported():
                    if window._latest_export_dir is not None:
                        event_loop.quit()

                poll = QtCore.QTimer()
                poll.setInterval(10)
                poll.timeout.connect(finish_when_exported)
                poll.start()
                QtCore.QTimer.singleShot(3000, event_loop.quit)
                event_loop.exec_()
                poll.stop()
                timer.stop()

            self.assertIsNotNone(window._latest_export_dir)
            self.assertIsNot(worker_threads[0], gui_thread)
            self.assertGreater(len(gui_ticks), 2)

        window.cloud_view.shutdown()
        window.deleteLater()


if __name__ == "__main__":
    unittest.main()
