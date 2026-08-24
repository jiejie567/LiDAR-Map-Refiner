#!/usr/bin/env python3
"""Render the preregistered C1/C2 title decision as a hashed Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c1_root", type=Path)
    parser.add_argument("--c2-root", type=Path)
    parser.add_argument(
        "--pgo-report", type=Path,
        help="Frozen ICP-remeasured pose-graph ablation decision",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c1_root = args.c1_root.expanduser().resolve()
    c2_root = (
        args.c2_root.expanduser().resolve() if args.c2_root else c1_root
    )
    c1_path = c1_root / "c1_source_data.json"
    c2_path = c2_root / "c2_source_data.json"
    c1 = json.loads(c1_path.read_text())
    c2 = json.loads(c2_path.read_text()) if c2_path.is_file() else None
    pgo_path = (
        args.pgo_report.expanduser().resolve() if args.pgo_report else None
    )
    pgo = (
        json.loads(pgo_path.read_text(encoding="utf-8"))
        if pgo_path and pgo_path.is_file() else None
    )
    c1_pass = bool(c1["c1"]["passed"])
    c2_pass = bool(c2 and c2.get("passed"))
    if c1_pass:
        decision = "PROCEED: GhostLoop — auditing and recovering"
    elif c2_pass:
        decision = "PROCEED: LoopAudit — recovery demoted to a limitation"
    else:
        decision = "STOP ICRA SUBMISSION: neither preregistered contribution passed"

    lines = [
        "# GhostLoop / LoopAudit ICRA 2027 Go–No-Go",
        "",
        f"**Decision: {decision}.**",
        "",
        "This report is generated only from hash-verified manifests and proposal ledgers.",
        "",
        "## C1 — map-inconsistency recovery",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {mark(value)} |"
        for name, value in c1["c1"]["gates"].items()
    )
    lines += [
        "",
        f"Retained map proposals: {c1['c1']['retained_map_recovery_loop_count']}; "
        f"datasets: {', '.join(c1['c1']['recovering_datasets']) or 'none'}.",
        "",
        "### Every retained/retracted strict map proposal with truth",
        "",
        "| Dataset | Pair | Decision | Fixed separation (cm) | Truth error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in c1["proposal_rows"]:
        if row["source"] != "map_inconsistency" or not row["gt_available"]:
            continue
        before = row.get("before_separation_m")
        after = row.get("after_separation_m")
        separation = (
            "—" if before is None or after is None
            else f"{100 * before:.1f} → {100 * after:.1f}"
        )
        trans = row.get("relative_translation_error_m")
        rot = row.get("relative_rotation_error_deg")
        truth = (
            "—" if trans is None else f"{trans:.2f} m"
            + ("" if rot is None else f", {rot:.1f}°")
        )
        lines.append(
            f"| {row['dataset']} | {row['target_id']}→{row['source_id']} | "
            f"{row['decision']} | {separation} | {truth} |"
        )

    lines += ["", "## C2 — source-agnostic loop audit", ""]
    if c2 is None:
        lines.append("C2 evidence is not frozen yet.")
    else:
        lines += ["| Data/method gate | Result |", "|---|---:|"]
        for group in ("c2_data_gate", "c2_method_gate"):
            lines.extend(
                f"| {name.replace('_', ' ')} | {mark(value)} |"
                for name, value in c2[group].items()
            )
        lines += [
            "",
            "| Policy | Accepted | False accepted | Recall | False-accept rate | >5 m |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, item in c2["policies"].items():
            recall = "—" if item["recall"] is None else f"{item['recall']:.3f}"
            far = (
                "—" if item["false_accept_rate"] is None
                else f"{item['false_accept_rate']:.3f}"
            )
            lines.append(
                f"| {name} | {item['accepted']} | {item['false_accepted']} | "
                f"{recall} | {far} | {item['catastrophic_over_5m']} |"
            )

    lines += ["", "## ICP-remeasured PGO factor audit", ""]
    if pgo is None:
        lines.append("PGO ablation evidence is not frozen yet.")
    else:
        lines += ["| Promotion gate | Result |", "|---|---:|"]
        lines.extend(
            f"| {name.replace('_', ' ')} | {mark(value)} |"
            for name, value in pgo["gates"].items()
        )
        replacement = pgo["sequential_measurement_replacement"]
        reweight = pgo["sequential_information_reweighting"]
        lines += [
            "",
            f"Replacing sequential measurements: original ATE "
            f"{replacement['original_graph_ate_m']:.3f} m; best replacement "
            f"{min(item['ate_m'] for item in replacement['variants'].values()):.3f} m.",
            f"Sequential reweighting versus loop-only: "
            f"{reweight['oriented_reweight_ate_m']:.3f} m versus "
            f"{reweight['loop_only_oriented_ate_m']:.3f} m.",
            f"Decision: **{pgo['recommended_role']}**.",
        ]

    lines += [
        "",
        "## Evidence hashes",
        "",
        f"- C1 source data: `{hashlib.sha256(c1_path.read_bytes()).hexdigest()}`",
    ]
    if c2_path.is_file():
        lines.append(
            f"- C2 source data: `{hashlib.sha256(c2_path.read_bytes()).hexdigest()}`"
        )
    if pgo_path and pgo_path.is_file():
        lines.append(
            f"- PGO ablation: `{hashlib.sha256(pgo_path.read_bytes()).hexdigest()}`"
        )
    output = (
        args.output.expanduser().resolve()
        if args.output else c2_root / "GO_NO_GO.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(decision)


if __name__ == "__main__":
    main()
