import sys
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ghostloop_eval"
sys.path.insert(0, str(SCRIPT_DIR))

from c2_from_ledger import (  # noqa: E402
    comparison_metrics,
    descriptor_proposal_keys,
    metrics,
    proposal_key,
)
from proposal_ledger import ProposalLedger  # noqa: E402


class C2EvidenceTest(unittest.TestCase):
    def test_proposal_key_keeps_dataset_and_directed_pair(self):
        row = {"dataset": "ntu", "target_id": "12", "source_id": 84}
        self.assertEqual(proposal_key(row), ("ntu", 12, 84))

    def test_development_overlap_includes_failed_registration_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "ntu"
            ledger_path = session / "proposal_ledgers" / "run.jsonl"
            ledger = ProposalLedger(ledger_path, "run")
            ledger.append({
                "source": "stage0_descriptor",
                "round": 0,
                "source_id": 84,
                "target_id": 12,
                "proposal_id": "failed-registration",
                "decision": "rejected",
                "reason": "registration failed",
            })
            (session / "auto_repair_summary.json").write_text(
                json.dumps({"proposal_ledger": {"path": str(ledger_path)}}),
                encoding="utf-8",
            )
            self.assertEqual(
                descriptor_proposal_keys(root), {("ntu", 12, 84)}
            )

    def test_comparison_metrics_handles_novel_split_without_true_loops(self):
        rows = [
            {
                "gt_available": True,
                "truth_correct": False,
                "gicp_relative_translation_error_m": 2.0,
                "gicp_gate_passed": True,
                "gravity_gate_passed_1deg": False,
            },
            {
                "gt_available": True,
                "truth_correct": False,
                "gicp_relative_translation_error_m": 3.0,
                "gicp_gate_passed": True,
                "gravity_gate_passed_1deg": True,
            },
        ]
        policies = {
            "gicp_only": metrics(rows, lambda row: row["gicp_gate_passed"]),
            "full_audit": metrics(
                rows, lambda row: row["gravity_gate_passed_1deg"]
            ),
        }
        comparison = comparison_metrics(policies)
        self.assertAlmostEqual(
            comparison["false_accept_reduction_vs_gicp"], 0.5
        )
        self.assertIsNone(comparison["recall_drop_percentage_points"])


if __name__ == "__main__":
    unittest.main()
