import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ghostloop_eval"
GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
OPT_DIR = GUI_DIR / "manual_loop_closure" / "python_optimizer"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(GUI_DIR))
sys.path.insert(0, str(OPT_DIR))

from causal_audit import (  # noqa: E402
    FrozenGhostEvidence,
    evaluate_causal_effect,
    evaluate_graph_trial,
    factor_consistency,
    gravity_consistency,
    measure_frozen_evidence,
)
from proposal_ledger import ProposalLedger, proposal_id  # noqa: E402


class CausalProposalAuditTest(unittest.TestCase):
    def _evidence(self) -> FrozenGhostEvidence:
        return FrozenGhostEvidence(
            local_xyz=np.asarray([
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.12, 0.0, 0.0],
                [0.12, 1.0, 0.0],
            ]),
            pose_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
            layer_low=np.asarray([True, True, False, False]),
            voxel_ids=np.zeros(4, dtype=np.int32),
            voxel_normals=np.asarray([[1.0, 0.0, 0.0]]),
            voxel_keys=np.asarray([[0, 0, 0]], dtype=np.int64),
            reference_separation_m=0.12,
            reported_separation_m=0.12,
        )

    def test_fixed_identity_is_reprojected_without_relabeling(self):
        evidence = self._evidence()
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        self.assertAlmostEqual(measure_frozen_evidence(evidence, poses), 0.12)
        poses[1, 0, 3] = -0.08
        self.assertAlmostEqual(measure_frozen_evidence(evidence, poses), 0.04)

    def test_effect_gate_matches_preregistered_thresholds(self):
        closed = evaluate_causal_effect(0.126, 0.079)
        self.assertTrue(closed.retained)
        substantial = evaluate_causal_effect(0.14, 0.095)
        self.assertTrue(substantial.retained)
        weak = evaluate_causal_effect(0.11, 0.09)
        self.assertFalse(weak.retained)
        worse = evaluate_causal_effect(0.09, 0.10)
        self.assertFalse(worse.retained)
        self.assertIn("worsened", worse.reason)

    def test_trial_graph_gate_retracts_local_and_global_inconsistency(self):
        clean = evaluate_graph_trial(0.1, 0.2, 0.18, residual_limit_m=1.5)
        self.assertTrue(clean.retained)
        local_bad = evaluate_graph_trial(1.6, 1.6, 0.2, residual_limit_m=1.5)
        self.assertFalse(local_bad.retained)
        self.assertIn("candidate factor", local_bad.reason)
        global_bad = evaluate_graph_trial(0.2, 1.7, 0.2, residual_limit_m=1.5)
        self.assertFalse(global_bad.retained)
        self.assertIn("global residual", global_bad.reason)
        inherited = evaluate_graph_trial(0.2, 1.7, 1.65, residual_limit_m=1.5)
        self.assertTrue(inherited.retained)

    def test_factor_consistency_uses_anisotropic_information(self):
        measured = np.eye(4)
        solved = np.eye(4)
        solved[0, 3] = 0.1
        strong_x = np.diag([10000.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weak_x = np.diag([1.0, 10000.0, 1.0, 1.0, 1.0, 1.0])
        rejected = factor_consistency(measured, solved, strong_x)
        retained = factor_consistency(measured, solved, weak_x)
        self.assertGreater(rejected.normalized_score, 1.0)
        self.assertLess(retained.normalized_score, 1.0)
        self.assertAlmostEqual(rejected.translation_residual_m, 0.1)

    def test_gravity_consistency_uses_target_to_source_rotation(self):
        source_up = np.asarray([0.0, 1.0, 0.0])
        target_up = np.asarray([0.0, 0.0, 1.0])
        measured = np.eye(4)
        measured[:3, :3] = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        valid = gravity_consistency(
            measured, source_up, target_up, max_error_deg=1.0
        )
        self.assertTrue(valid.available)
        self.assertTrue(valid.passed)
        self.assertLess(valid.error_deg, 1e-8)
        reversed_measurement = np.linalg.inv(measured)
        invalid = gravity_consistency(
            reversed_measurement, source_up, target_up, max_error_deg=1.0
        )
        self.assertFalse(invalid.passed)
        self.assertGreater(invalid.error_deg, 80.0)

    def test_gravity_consistency_is_yaw_invariant(self):
        angle = np.deg2rad(137.0)
        measured = np.eye(4)
        measured[:3, :3] = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        effect = gravity_consistency(
            measured, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
        )
        self.assertTrue(effect.passed)
        self.assertLess(effect.error_deg, 1e-8)

    def test_frozen_evidence_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.npz"
            evidence = self._evidence()
            evidence.save(path)
            restored = FrozenGhostEvidence.load(path)
            np.testing.assert_array_equal(restored.local_xyz, evidence.local_xyz)
            np.testing.assert_array_equal(restored.layer_low, evidence.layer_low)
            self.assertEqual(restored.reference_separation_m, 0.12)

    def test_ledger_is_append_only_and_hash_chained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            ledger = ProposalLedger(path, "run-1")
            pid = proposal_id("run-1", "map", 1, 10, 2, "region-1")
            ledger.append({"proposal_id": pid, "decision": "retained"})
            ledger.append({"proposal_id": "p2", "decision": "rejected"})
            records = ProposalLedger.verify(path)
            self.assertEqual([item["decision"] for item in records], ["retained", "rejected"])
            with self.assertRaises(FileExistsError):
                ProposalLedger(path, "run-2")
            lines = path.read_text().splitlines()
            damaged = json.loads(lines[0])
            damaged["decision"] = "changed"
            lines[0] = json.dumps(damaged)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid ledger hash"):
                ProposalLedger.verify(path)


if __name__ == "__main__":
    unittest.main()
