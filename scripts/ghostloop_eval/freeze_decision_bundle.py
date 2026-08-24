#!/usr/bin/env python3
"""Hash the frozen C1/C2 decision report, source data, figures, and code."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CODE_FILES = (
    "auto_repair_headless.py",
    "causal_audit.py",
    "proposal_ledger.py",
    "evaluate_icra2027_rebuild.py",
    "c2_from_ledger.py",
    "run_c2_baselines.py",
    "write_go_no_go.py",
    "plot_go_no_go_evidence.py",
    "summarize_pgo_innovation_ablation.py",
    "freeze_decision_bundle.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c1_root", type=Path)
    parser.add_argument("c2_root", type=Path)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--pgo-report", type=Path)
    args = parser.parse_args()
    c1_root = args.c1_root.expanduser().resolve()
    c2_root = args.c2_root.expanduser().resolve()
    bundle = args.bundle_root.expanduser().resolve()
    baseline_manifest = (
        args.baseline_manifest.expanduser().resolve()
        if args.baseline_manifest else c2_root / "c2_baseline_manifest.json"
    )
    pgo_report = (
        args.pgo_report.expanduser().resolve()
        if args.pgo_report else bundle / "pgo_innovation_decision.json"
    )
    report = bundle / "GO_NO_GO.md"
    c1 = c1_root / "c1_source_data.json"
    c2 = c2_root / "c2_source_data.json"
    decision = "STOP ICRA SUBMISSION"
    if report.is_file():
        first_lines = report.read_text(encoding="utf-8").splitlines()[:5]
        for line in first_lines:
            if "Decision:" in line:
                decision = line.replace("**Decision:", "").strip(" *. ")
                break
    required = [
        c1,
        c1_root / "proposal_source_data.csv",
        c1_root / "c1_evidence_manifest.json",
        c2,
        c2_root / "c2_candidate_source_data.csv",
        c2_root / "c2_evidence_manifest.json",
        baseline_manifest,
        pgo_report,
        report,
        report.with_suffix(report.suffix + ".sha256"),
        bundle / "figure3/go_no_go_figure_source_data.csv",
        bundle / "figure3/go_no_go_evidence.svg",
        bundle / "figure3/go_no_go_evidence.pdf",
        bundle / "figure3/go_no_go_evidence.png",
        bundle / "figure3/go_no_go_evidence.tiff",
        bundle / "figure3/go_no_go_evidence_qa.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen evidence: " + ", ".join(missing))
    c1_data = json.loads(c1.read_text(encoding="utf-8"))
    c2_data = json.loads(c2.read_text(encoding="utf-8"))
    pgo_data = json.loads(pgo_report.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "gate_results": {
            "c1_passed": bool(c1_data["c1"]["passed"]),
            "c2_passed": bool(c2_data["passed"]),
            "pgo_promoted": bool(pgo_data["promote_as_paper_contribution"]),
        },
        "truth_protocol": c1_data["loop_truth_protocol"],
        "evidence_files": [file_record(path) for path in required],
        "code_files": [file_record(HERE / name) for name in CODE_FILES],
    }
    output = bundle / "decision_bundle_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest["gate_results"], indent=2))


if __name__ == "__main__":
    main()
