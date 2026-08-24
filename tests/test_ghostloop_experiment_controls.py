import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ghostloop_eval"
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_io import (  # noqa: E402
    CONSTRAINT_HEADER,
    experiment_controls_payload,
    load_balm_root_voxel_size,
    load_initial_constraints_csv,
    sha256_file,
    validate_round_inputs,
    voxel_mask,
)


class GhostLoopExperimentControlsTest(unittest.TestCase):
    def test_headless_root_override_reaches_balm_and_report(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(encoding="utf-8")
        self.assertIn("root_voxel_size=BALM_ROOT_VOXEL_SIZE", source)
        self.assertIn("'--root-voxel', str(balm_params.root_voxel_size)", source)
        self.assertIn("'experiment_controls': experiment_controls_payload(", source)

    def test_headless_public_entry_is_production_only(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("choices=('production',)", source)
        self.assertNotIn("choices=('production', 'ghost'", source)
        rebuild = (SCRIPT_DIR / "icra2027_rebuild.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=("production",)', rebuild)

    def test_gui_has_no_diagnosis_driven_loop_controls(self):
        gui_source = (
            SCRIPT_DIR.parents[1] / "gui" / "manual_loop_closure_tool.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Ghost Loop (Legacy)", gui_source)
        self.assertNotIn("Ghost Sug. (Legacy)", gui_source)
        self.assertNotIn("def toggle_auto_iterate", gui_source)
        self.assertNotIn("def apply_balm_suggestions", gui_source)

    def test_production_is_default_and_runs_one_terminal_balm(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("default='production'", source)
        self.assertIn("if MODE == 'production':\n    baseline_regions = []", source)
        self.assertIn(
            "if MODE == 'production':\n    descriptors, masks = build_descriptor_stack(",
            source,
        )
        self.assertIn("run_balm_cli(out, '_balm_final')", source)
        self.assertIn("termination_policy=(", source)
        self.assertIn("'single_final_double_sided_balm'", source)

    def test_headless_publishes_atomic_gui_result_contract(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'--result-json'", source)
        self.assertIn("'lidar-map-refiner/headless-result'", source)
        self.assertIn("temporary.replace(RESULT_JSON)", source)
        self.assertIn("write_result_contract(\n    out,", source)
        self.assertLess(
            source.index("write_repair_summary(\n    'complete'"),
            source.index("write_result_contract(\n    out,"),
        )

    def test_calibrated_pgo_profile_is_explicit_and_legacy_remains_available(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("choices=('legacy', 'calibrated')", source)
        self.assertIn("PRODUCTION_PGO_PROFILE['optimize_mode']", source)
        self.assertIn("PRODUCTION_PGO_PROFILE['manual_information_scale']", source)
        self.assertIn("PRODUCTION_PGO_PROFILE['rotation_sigma_deg']", source)
        self.assertIn("PRODUCTION_PGO_PROFILE['cluster_information_budget']", source)
        self.assertIn("PGO_OPTIMIZE_MODE = 'isam2'", source)
        self.assertIn("'--manual-information-scale'", source)
        self.assertIn("'correlation_clustering': CORRELATION_CLUSTERING_METHOD", source)

    def test_rebuild_freezes_pgo_profile_and_weighting_code(self):
        source = (SCRIPT_DIR / "icra2027_rebuild.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('default="calibrated"', source)
        self.assertIn('default="production"', source)
        self.assertIn('repair_mode,\n        environment', source)
        self.assertIn('"--pgo-profile", pgo_profile', source)
        self.assertIn('PACKAGE_ROOT.rglob("*.py")', source)

    def test_dead_multiframe_probe_is_not_in_production_path(self):
        source = (SCRIPT_DIR / "auto_repair_headless.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("multiframe_moved(", source)
        self.assertNotIn("GHOSTLOOP_FINAL_MAX_MOVE", source)

    def test_fixed_stage0_load_preserves_rows_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constraints.csv"
            rows = [
                ["1", "9", "2", *("0.125000" for _ in range(13))],
                ["1", "8", "1", *("-0.250000" for _ in range(13))],
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(CONSTRAINT_HEADER)
                writer.writerows(rows)
            before = sha256_file(path)
            loaded = load_initial_constraints_csv(path, pose_count=10)
            self.assertEqual(loaded, rows)
            self.assertEqual(sha256_file(path), before)

    def test_root_voxel_is_loaded_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "balm_report.json"
            report.write_text(
                json.dumps({"params": {"root_voxel_size": 5.0}}),
                encoding="utf-8",
            )
            self.assertEqual(load_balm_root_voxel_size(report), 5.0)
            payload = experiment_controls_payload(5.0, None, 0)
            self.assertEqual(payload["balm_root_voxel_size"], 5.0)
            self.assertEqual(payload["stage0_source"], "retrieval")

    def test_non_one_meter_voxel_selection(self):
        points = np.asarray([[19.9, 0.1, -0.1], [20.1, 0.1, -0.1], [24.9, 4.9, -0.1]])
        selected = voxel_mask(points, [(4, 0, -1)], root_voxel_size=5.0)
        np.testing.assert_array_equal(selected, [False, True, True])

    def test_arbitrary_round_counts(self):
        validate_round_inputs(
            [Path("r1"), Path("r2"), Path("r3")],
            ["R1", "R2", "R3"],
            [5, 7, 7],
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            validate_round_inputs([Path("r1")], ["R1", "R2"], [5])


if __name__ == "__main__":
    unittest.main()
