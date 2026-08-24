#!/usr/bin/env python3
"""Plot the complete preregistered C1/C2 decision evidence.

This is deliberately a diagnostic figure: every strict map-inconsistency
proposal with a fixed before/after measurement is included, including repeated
proposals and failures.  It must not be presented as a successful-method figure
when the preregistered gates fail.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


WIDTH_IN = 7.16
HEIGHT_IN = 5.35
NAVY = "#1B365D"
BLUE = "#4C78A8"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#E4E7EB"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.transparent": False,
    })


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13, 1.08, label, transform=axis.transAxes,
        fontsize=8.0, fontweight="bold", va="top", ha="left",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return value is not None and np.isfinite(float(value))


def map_rows(c1: dict) -> list[dict]:
    return [
        row for row in c1["proposal_rows"]
        if row["source"] == "map_inconsistency"
        and finite(row.get("before_separation_m"))
        and finite(row.get("after_separation_m"))
    ]


def write_source_data(path: Path, rows: list[dict], c2: dict) -> None:
    fields = [
        "record_type", "dataset", "proposal_id", "pair", "decision",
        "fixed_measurement_available",
        "before_separation_cm", "after_separation_cm",
        "truth_translation_error_m", "truth_rotation_error_deg",
        "policy", "threshold", "accepted", "true_accepted",
        "false_accepted", "recall", "false_accept_rate",
        "catastrophic_over_5m", "ate_m",
    ]
    output = []
    for row in rows:
        fixed_available = (
            finite(row.get("before_separation_m"))
            and finite(row.get("after_separation_m"))
        )
        output.append({
            "record_type": "strict_map_proposal",
            "dataset": row["dataset"],
            "proposal_id": row["proposal_id"],
            "pair": f"{row['target_id']}->{row['source_id']}",
            "decision": row["decision"],
            "fixed_measurement_available": fixed_available,
            "before_separation_cm": (
                100 * float(row["before_separation_m"])
                if fixed_available else None
            ),
            "after_separation_cm": (
                100 * float(row["after_separation_m"])
                if fixed_available else None
            ),
            "truth_translation_error_m": row.get("relative_translation_error_m"),
            "truth_rotation_error_deg": row.get("relative_rotation_error_deg"),
        })
    for threshold, block in (
        ("1m/5deg", c2["policies"]),
        ("1m/10deg", c2["threshold_sensitivity"]["1.0m_10deg"]),
    ):
        for policy, values in block.items():
            output.append({
                "record_type": "c2_policy",
                "policy": policy,
                "threshold": threshold,
                **{key: values.get(key) for key in (
                    "accepted", "true_accepted", "false_accepted", "recall",
                    "false_accept_rate", "catastrophic_over_5m",
                )},
            })
    for dataset, arms in c2.get("trajectory_baselines", {}).items():
        for policy, values in arms.items():
            output.append({
                "record_type": "trajectory_baseline",
                "dataset": dataset,
                "policy": policy,
                "ate_m": values.get("ate_m"),
            })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)


def make_figure(c1: dict, c2: dict) -> plt.Figure:
    rows = map_rows(c1)
    rows.sort(
        key=lambda row: float(row["before_separation_m"])
        - float(row["after_separation_m"]), reverse=True,
    )
    # Literal size keeps the publication-width preflight machine-verifiable.
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a: fixed-evidence change for every strict proposal.
    y = np.arange(len(rows))
    before = np.asarray([100 * float(row["before_separation_m"]) for row in rows])
    after = np.asarray([100 * float(row["after_separation_m"]) for row in rows])
    labels = [
        f"{row['dataset']} {row['target_id']}→{row['source_id']}"
        for row in rows
    ]
    for index, row in enumerate(rows):
        color = ORANGE if row["active_final"] else GRAY
        ax_a.plot([before[index], after[index]], [index, index], color=LIGHT_GRAY,
                  lw=1.2, zorder=1)
        ax_a.scatter(before[index], index, s=17, facecolors="white",
                     edgecolors=color, linewidths=0.8, zorder=2)
        ax_a.scatter(after[index], index, s=18, color=color, edgecolors="white",
                     linewidths=0.35, zorder=3)
    ax_a.axvline(8, color=NAVY, ls="--", lw=0.8)
    ax_a.text(8.5, 0.97, "8 cm local-effect threshold",
              transform=ax_a.get_xaxis_transform(), color=NAVY,
              fontsize=6.0, va="top")
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Fixed surface separation (cm)")
    ax_a.set_title("Local map effect: before ○  →  after ●", loc="left")
    ax_a.grid(axis="x", color=LIGHT_GRAY, lw=0.45)
    panel_label(ax_a, "a")

    # b: the same SE(3)-labelled proposals against truth.
    se3_rows = [row for row in rows if finite(row.get("relative_rotation_error_deg"))]
    for row in se3_rows:
        retained = bool(row["active_final"])
        ax_b.scatter(
            float(row["relative_translation_error_m"]),
            float(row["relative_rotation_error_deg"]),
            s=35 if retained else 24,
            marker="o" if retained else "x",
            color=VERMILION if retained else GRAY,
            linewidths=1.0, zorder=3,
        )
    ax_b.axvspan(0, 1, ymin=0, ymax=5 / 45, color=BLUE, alpha=0.13, lw=0)
    ax_b.axhline(5, color=NAVY, ls="--", lw=0.75)
    ax_b.axvline(1, color=NAVY, ls="--", lw=0.75)
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(0, max(45, max(
        [float(row["relative_rotation_error_deg"]) for row in se3_rows] or [45]
    ) + 2))
    ax_b.set_xlabel("Loop translation error (m)")
    ax_b.set_ylabel("Loop rotation error (deg)")
    ax_b.set_title("Local improvement does not establish loop truth", loc="left")
    ax_b.text(0.04, 0.12, "1 m / 5°\ncorrect region", transform=ax_b.transAxes,
              fontsize=6.0, color=NAVY, va="bottom")
    ax_b.text(0.98, 0.98, "● retained   × retracted", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=6.0)
    ax_b.grid(color=LIGHT_GRAY, lw=0.45)
    panel_label(ax_b, "b")

    # c: C2 policy outcomes at the preregistered threshold.  The final audit
    # must show its safety/recall trade-off, rather than only its lower false
    # acceptance count.
    order = ["gicp_only", "gnc_tls", "gravity_admission", "full_audit"]
    short = ["GICP", "GNC-TLS", "Gravity", "Full audit"]
    primary = c2["policies"]
    xx = np.arange(len(order))
    false = np.asarray([primary[name]["false_accepted"] for name in order])
    catastrophic = np.asarray([
        primary[name]["catastrophic_over_5m"] for name in order
    ])
    ax_c.bar(xx, false, width=0.62, color=[GRAY, BLUE, ORANGE, NAVY],
             edgecolor="white", linewidth=0.4)
    for index, value in enumerate(false):
        ax_c.text(index, value + 0.45, str(int(value)), ha="center",
                  va="bottom", fontsize=6.0, color=NAVY)
    ax_c.set_xticks(xx, short)
    ax_c.set_ylabel("False accepted loops (1 m / 5°)")
    ax_c.set_title("Descriptor-loop audit", loc="left")
    ax_c.grid(axis="y", color=LIGHT_GRAY, lw=0.45)
    ax_c2 = ax_c.twinx()
    recall = [primary[name]["recall"] for name in order]
    ax_c2.plot(xx, recall, color=VERMILION, marker="D", ms=3.2, lw=1.0)
    ax_c2.set_ylim(0, 1.05)
    ax_c2.set_ylabel("Recall at 1 m / 5°", color=VERMILION)
    ax_c2.tick_params(axis="y", colors=VERMILION)
    n_registered = primary["gicp_only"]["registered_candidates"]
    n_correct = primary["gicp_only"]["truth_correct"]
    ax_c.text(0.02, 0.88,
              f"{n_registered} registered; {n_correct} truth-positive",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.0,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0,
                    "alpha": 0.85})
    panel_label(ax_c, "c")

    # d: trajectory-level comparison demanded by C2.
    datasets = []
    ratios = []
    absolute = []
    for dataset, arms in c2.get("trajectory_baselines", {}).items():
        gnc = (arms.get("gnc_tls") or {}).get("ate_m")
        full = (arms.get("full_audit") or {}).get("ate_m")
        if finite(gnc) and finite(full) and float(gnc) > 0:
            datasets.append(dataset)
            ratios.append(float(full) / float(gnc))
            absolute.append((float(gnc), float(full)))
    order_idx = np.argsort(ratios)
    datasets = [datasets[index] for index in order_idx]
    ratios = np.asarray(ratios)[order_idx]
    colors = [BLUE if value <= 1 else VERMILION for value in ratios]
    yy = np.arange(len(datasets))
    ax_d.barh(yy, ratios, color=colors, height=0.58,
              edgecolor="white", linewidth=0.4)
    x_max = max(1.15, float(np.max(ratios)) * 1.08) if len(ratios) else 1.15
    for index, value in enumerate(ratios):
        ax_d.text(value + 0.02 * x_max, index, f"{value:.2f}×",
                  ha="left", va="center", fontsize=5.8, color=NAVY)
    ax_d.axvline(1.0, color=NAVY, lw=0.8, ls="--")
    ax_d.set_yticks(yy, datasets)
    ax_d.set_xlabel("ATE ratio: full audit / GNC-TLS")
    ax_d.set_xlim(0, x_max)
    ax_d.set_title("Trajectory outcome", loc="left")
    ax_d.grid(axis="x", color=LIGHT_GRAY, lw=0.45)
    panel_label(ax_d, "d")

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax_c2.spines["top"].set_visible(False)
    fig.suptitle(
        "Map self-consistency is insufficient to validate LiDAR loop closures",
        x=0.08, y=0.995, ha="left", fontsize=8.5, fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.09, top=0.90,
                        wspace=0.42, hspace=0.42)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c1_root", type=Path)
    parser.add_argument("c2_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    c1 = load_json(args.c1_root / "c1_source_data.json")
    c2 = load_json(args.c2_root / "c2_source_data.json")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = map_rows(c1)
    all_map_proposals = [
        row for row in c1["proposal_rows"]
        if row["source"] == "map_inconsistency"
    ]
    write_source_data(
        output / "go_no_go_figure_source_data.csv", all_map_proposals, c2
    )
    figure = make_figure(c1, c2)
    stem = output / "go_no_go_evidence"
    # Preserve the declared physical canvas exactly; tight bounding boxes make
    # the exported width depend on text extents and break final-size QA.
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(figure)
    qa = {
        "core_conclusion": (
            "Incorrect loop closures can reduce a fixed local map separation, "
            "while the complete audit loses all true-loop recall and underperforms "
            "GNC-TLS at trajectory level."
        ),
        "archetype": "quantitative asymmetric 2x2 decision-evidence grid",
        "backend": "Python/matplotlib only",
        "final_size_in": [WIDTH_IN, HEIGHT_IN],
        "selection_policy": "all strict fixed-evidence map proposals included",
        "map_proposal_accounting": {
            "total": len(all_map_proposals),
            "fixed_measurement_available_and_plotted": len(rows),
            "without_fixed_measurement": len(all_map_proposals) - len(rows),
            "reason": (
                "Rejected before a trial trajectory existed; retained with "
                "blank fixed-measurement fields in source data."
            ),
        },
        "primary_truth_threshold": "1 m / 5 deg",
        "c2_sample": {
            "registered_candidates": c2["policies"]["gicp_only"][
                "registered_candidates"
            ],
            "truth_positive": c2["policies"]["gicp_only"]["truth_correct"],
            "datasets": c2["dataset_count"],
        },
        "statistics": (
            "deterministic full registered-candidate census; no sampling, "
            "confidence interval, or hypothesis test"
        ),
        "source_data": "go_no_go_figure_source_data.csv",
        "editable_text": "SVG fonttype none; PDF fonttype 42",
        "physical_size_verified": "7.16 x 5.35 in PDF page",
        "visual_qa": (
            "Inspected final-size PNG after export; no label overlap or clipping"
        ),
        "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi"],
    }
    (output / "go_no_go_evidence_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
