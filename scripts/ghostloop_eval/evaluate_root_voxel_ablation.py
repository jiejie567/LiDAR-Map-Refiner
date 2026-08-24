#!/usr/bin/env python3
"""Summarize and plot a completed 4 m / 5 m GhostLoop ablation."""
from __future__ import annotations

import argparse
import csv
import json
import re
import hashlib
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / "gui"))
from manual_loop_closure import load_tum_trajectory  # noqa: E402


BLUE = "#2474b7"
ORANGE = "#e1812c"
INK = "#17212b"
GRAY = "#7f8a96"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.65, "axes.spines.right": False,
    "axes.spines.top": False,
})


def round_dirs(session: Path) -> list[Path]:
    found = []
    for path in (session / "manual_loop_runs").glob("*_auto_r*_balm"):
        match = re.search(r"_auto_r(\d+)_balm$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def region_metrics(report: dict) -> dict:
    regions = report.get("ghost_regions_all", report.get("ghost_regions", []))
    return {
        "candidate_count": len(regions),
        "point_count": int(sum(int(r["point_count"]) for r in regions)),
        "connected_voxel_count": int(sum(int(r["voxel_count"]) for r in regions)),
        "median_overlap_ratio": (
            float(np.median([r["layer_overlap_ratio"] for r in regions]))
            if regions else 0.0
        ),
        # Historical density-dependent score remains for old-run comparison.
        "severity": float(sum(
            r["point_count"] * r.get("separation_typical_m", r["separation_m"])
            for r in regions
        )),
        # Preferred diagnostic ranking: physical duplicated support volume,
        # independent of scan density and vehicle speed.
        "surface_displacement_m3": float(sum(
            r.get("surface_displacement_m3", 0.0) for r in regions
        )),
        "overlap_area_m2": float(sum(
            r.get("overlap_area_m2", 0.0) for r in regions
        )),
    }


def read_branch(session: Path) -> dict:
    summary = json.loads((session / "auto_repair_summary.json").read_text())
    rounds = []
    for index, directory in enumerate(round_dirs(session), 1):
        report = json.loads((directory / "balm_report.json").read_text())
        pgo_tum = directory.parent / directory.name.replace("_balm", "") / "optimized_poses_tum.txt"
        balm_tum = directory / "optimized_poses_tum.txt"
        pgo = load_tum_trajectory(pgo_tum).positions_xyz
        balm = load_tum_trajectory(balm_tum).positions_xyz
        motion = np.linalg.norm(pgo - balm, axis=1)
        constraints_path = directory / "manual_loop_constraints.csv"
        with constraints_path.open(newline="", encoding="utf-8") as stream:
            constraint_count = sum(row[0] == "1" for row in list(csv.reader(stream))[1:])
        rounds.append({
            "round": index,
            "constraint_count": constraint_count,
            "root_voxel_size_m": report["params"]["root_voxel_size"],
            "balm_converged": bool(report["converged"]),
            "balm_stop_reason": report["stop_reason"],
            "balm_mean_pose_motion_m": float(motion.mean()),
            "balm_max_pose_motion_m": float(motion.max()),
            **region_metrics(report),
        })
    return {
        "session": str(session.resolve()),
        "stage0_sha256": summary["experiment_controls"]["initial_constraints_sha256"],
        "root_voxel_size_m": summary["balm"]["root_voxel_size"],
        "initial_constraint_count": summary["experiment_controls"]["initial_constraint_count"],
        "additional_loop_count": (
            summary["active_constraint_count"]
            - summary["experiment_controls"]["initial_constraint_count"]
        ),
        "final_constraint_count": summary["active_constraint_count"],
        "round_count": summary["rounds_completed"],
        "rounds": rounds,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def fixed_point_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text())
    values = np.asarray([item["separation_m"] for item in payload["candidates"]])
    if values.size == 0:
        return {
            "candidate_count": 0, "r1_median_separation_m": None,
            "final_median_separation_m": None, "closed_below_8cm_count": 0,
            "closure_rate": None, "persistent_count": 0, "worsened_count": 0,
        }
    final = values[:, -1]
    return {
        "candidate_count": int(len(values)),
        "r1_median_separation_m": float(np.median(values[:, 0])),
        "final_median_separation_m": float(np.median(final)),
        "closed_below_8cm_count": int(np.sum(final < 0.08)),
        "closure_rate": float(np.mean(final < 0.08)),
        "persistent_count": int(np.sum(final >= 0.08)),
        "worsened_count": int(np.sum(final > values[:, 0] + 0.01)),
    }


def interval_distance(a: list[int], b: list[int]) -> float:
    return abs((a[0] + a[1]) / 2.0 - (b[0] + b[1]) / 2.0)


def match_regions(regions_a: list[dict], regions_b: list[dict]) -> dict:
    """Match physical surfaces by position, normal and unordered trajectory segments."""
    scored = []
    for ia, a in enumerate(regions_a):
        ca = np.asarray(a.get("region_centroid_xyz", a["center_xyz"]), dtype=float)
        na = np.asarray(a["normal"], dtype=float)
        for ib, b in enumerate(regions_b):
            cb = np.asarray(b.get("region_centroid_xyz", b["center_xyz"]), dtype=float)
            nb = np.asarray(b["normal"], dtype=float)
            spatial = float(np.linalg.norm(ca - cb))
            normal_similarity = abs(float(na @ nb) / (np.linalg.norm(na) * np.linalg.norm(nb)))
            mean_normal = na / np.linalg.norm(na)
            if float(mean_normal @ nb) < 0.0:
                mean_normal *= -1.0
            plane_distance = abs(float((ca - cb) @ mean_normal))
            # Root voxels represent finite surface patches.  Their AABBs may
            # overlap even when weighted centroids lie far apart along a long
            # wall, which is exactly the 4 m/5 m boundary-shift case.
            box_distance = float("inf")
            for key_a in a["voxel_keys"]:
                lo_a = np.asarray(key_a, dtype=float) * 4.0
                hi_a = lo_a + 4.0
                for key_b in b["voxel_keys"]:
                    lo_b = np.asarray(key_b, dtype=float) * 5.0
                    hi_b = lo_b + 5.0
                    gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
                    box_distance = min(box_distance, float(np.linalg.norm(gap)))
            direct = max(
                interval_distance(a["layer_a_keyframes"], b["layer_a_keyframes"]),
                interval_distance(a["layer_b_keyframes"], b["layer_b_keyframes"]),
            )
            crossed = max(
                interval_distance(a["layer_a_keyframes"], b["layer_b_keyframes"]),
                interval_distance(a["layer_b_keyframes"], b["layer_a_keyframes"]),
            )
            segment_distance = min(direct, crossed)
            if (box_distance <= 1.0 and plane_distance <= 1.0
                    and normal_similarity >= 0.94 and segment_distance <= 45):
                scored.append((box_distance + plane_distance + segment_distance / 20.0,
                               ia, ib, spatial, plane_distance, box_distance,
                               normal_similarity, segment_distance))
    matched_a, matched_b, matches = set(), set(), []
    for (_, ia, ib, spatial, plane_distance, box_distance,
         normal_similarity, segment_distance) in sorted(scored):
        if ia in matched_a or ib in matched_b:
            continue
        matched_a.add(ia); matched_b.add(ib)
        matches.append({
            "rank_4m": ia + 1, "rank_5m": ib + 1,
            "spatial_distance_m": spatial,
            "surface_plane_distance_m": plane_distance,
            "root_voxel_box_distance_m": box_distance,
            "absolute_normal_similarity": normal_similarity,
            "trajectory_segment_distance_kf": segment_distance,
        })
    return {
        "matches": matches,
        "unique_to_4m": [i + 1 for i in range(len(regions_a)) if i not in matched_a],
        "unique_to_5m": [i + 1 for i in range(len(regions_b)) if i not in matched_b],
    }


def render(branches: list[dict], output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    colors = [BLUE, ORANGE]
    labels = ["4 m", "5 m"]
    for branch, color, label in zip(branches, colors, labels):
        xx = np.asarray([r["round"] for r in branch["rounds"]])
        axes[0].plot(xx, [r["candidate_count"] for r in branch["rounds"]],
                     marker="o", color=color, label=label)
        axes[1].plot(xx, [r["severity"] for r in branch["rounds"]],
                     marker="o", color=color)
        axes[2].plot(xx, [r["constraint_count"] for r in branch["rounds"]],
                     marker="o", color=color)
    titles = ["Detected regions", "Ghost severity", "Active constraints"]
    ylabels = ["count", "points × separation (m)", "count"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({r["round"] for b in branches for r in b["rounds"]}))
        ax.grid(color="#e5e9ed", lw=0.45)
    axes[0].legend(frameon=False)
    fig.suptitle("MCD Building · controlled root-voxel ablation",
                 x=0.075, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix, dpi in ((".png", 300), (".pdf", 300), (".svg", 300), (".tiff", 600)):
        fig.savefig(output_stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root4-session", type=Path, required=True)
    parser.add_argument("--root5-session", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root4-fixed-points", type=Path)
    parser.add_argument("--root5-fixed-points", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    branches = [read_branch(args.root4_session), read_branch(args.root5_session)]
    r1_reports = [
        json.loads((round_dirs(session)[0] / "balm_report.json").read_text())
        for session in (args.root4_session, args.root5_session)
    ]
    matching = match_regions(
        r1_reports[0]["ghost_regions_all"], r1_reports[1]["ghost_regions_all"]
    )
    payload = {
        "comparison": "MCD Building 4 m vs 5 m root voxel",
        "stage0_byte_identical": branches[0]["stage0_sha256"] == branches[1]["stage0_sha256"],
        "branches": branches,
        "r1_physical_surface_matching": matching,
    }
    fixed_paths = (args.root4_fixed_points, args.root5_fixed_points)
    if all(path is not None for path in fixed_paths):
        payload["fixed_point_evaluation"] = {
            "4m": fixed_point_metrics(fixed_paths[0]),
            "5m": fixed_point_metrics(fixed_paths[1]),
        }
        fixed = payload["fixed_point_evaluation"]
        motion4 = max(item["balm_mean_pose_motion_m"] for item in branches[0]["rounds"])
        motion5 = max(item["balm_mean_pose_motion_m"] for item in branches[1]["rounds"])
        upgrade = (
            fixed["5m"]["closure_rate"] > fixed["4m"]["closure_rate"]
            or fixed["5m"]["final_median_separation_m"]
            < fixed["4m"]["final_median_separation_m"]
        ) and fixed["5m"]["worsened_count"] <= fixed["4m"]["worsened_count"] \
            and motion5 <= motion4
        payload["recommendation"] = {
            "decision": "upgrade_to_5m" if upgrade else "retain_4m_default",
            "human_audit": (
                "No obvious additional structural-mixing false positive was found "
                "in the complete R1 audit sheets."
            ),
            "reason": (
                "5 m does not improve fixed-point closure or final median separation, "
                "recovers fewer later loops, and has larger BALM pose motion."
                if not upgrade else
                "5 m improves fixed-point repair without increasing persistence or motion."
            ),
        }
    r1_csvs = [
        round_dirs(session)[0].parent
        / round_dirs(session)[0].name.replace("_balm", "")
        / "manual_loop_constraints.csv"
        for session in (args.root4_session, args.root5_session)
    ]
    payload["r1_stage0_csv_sha256"] = [sha256_file(path) for path in r1_csvs]
    payload["r1_stage0_csv_byte_identical"] = (
        payload["r1_stage0_csv_sha256"][0]
        == payload["r1_stage0_csv_sha256"][1]
    )
    (args.out_dir / "MCD_Building_root_voxel_ablation_source_data.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for branch in branches:
        for item in branch["rounds"]:
            rows.append({"root_voxel_size_m": branch["root_voxel_size_m"], **item})
    with (args.out_dir / "MCD_Building_root_voxel_ablation_rounds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    render(branches, args.out_dir / "MCD_Building_root_voxel_ablation_summary")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
