#!/usr/bin/env python3
"""Render every refined ghost candidate for human visual audit.

Each panel uses the same data-derived view: the horizontal axis is the fitted
surface normal (where duplication separates), while the vertical axis follows
the candidate component in the surface plane.  Nearby map points provide
structural context; the two temporally separated traversals are highlighted.

Usage:
  make_refined_ghost_audit.py <tum> <regions.json> <session> <output-dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / "gui"))
from manual_loop_closure import RegistrationWorkspace, load_tum_trajectory  # noqa: E402
from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
    BalmParams,
    prepare_local_cloud,
)
from experiment_io import load_balm_root_voxel_size, voxel_mask  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
    }
)

NEUTRAL = "#aab2bd"
BLUE = "#2474b7"
ORANGE = "#e1812c"
INK = "#17212b"
MAX_OBSERVATION_RANGE = 20.0


def _canonical(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    axis = int(np.argmax(np.abs(normal)))
    return -normal if normal[axis] < 0 else normal


def _frame_mask(ids: np.ndarray, interval: list[int]) -> np.ndarray:
    return (ids >= int(interval[0])) & (ids <= int(interval[1]))


def _view_basis(
    region: dict, root_voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = _canonical(np.asarray(region["normal"], dtype=float))
    keys = (np.asarray(region["voxel_keys"], dtype=float) + 0.5) * root_voxel_size
    centered = keys - keys.mean(axis=0)
    in_plane = centered - np.outer(centered @ normal, normal)
    if len(keys) > 1 and float(np.linalg.norm(in_plane)) > 0.25:
        _, _, vh = np.linalg.svd(in_plane, full_matrices=False)
        tangent = vh[0] - normal * float(vh[0] @ normal)
    else:
        vertical = np.array([0.0, 0.0, 1.0])
        tangent = vertical - normal * float(vertical @ normal)
        if np.linalg.norm(tangent) < 0.25:
            tangent = np.array([1.0, 0.0, 0.0])
            tangent -= normal * float(tangent @ normal)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    depth = np.cross(normal, tangent)
    depth /= max(float(np.linalg.norm(depth)), 1e-12)
    return normal, tangent, depth


def _sample(mask: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return indices
    # Deterministic sampling preserves reproducibility and prevents dense scans
    # from visually overwhelming the two traversals.
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def _draw_section(
    ax: plt.Axes,
    rank: int,
    region: dict,
    world: np.ndarray,
    pose_ids: np.ndarray,
    root_voxel_size: float,
) -> None:
    center = np.asarray(region.get("region_centroid_xyz", region["center_xyz"]), dtype=float)
    normal, tangent, depth = _view_basis(region, root_voxel_size)
    relative = world - center
    qn = relative @ normal
    qt = relative @ tangent
    qd = relative @ depth

    key_centers = (
        (np.asarray(region["voxel_keys"], dtype=float) + 0.5)
        * root_voxel_size - center
    )
    key_t = key_centers @ tangent
    t_lo = min(float(key_t.min()) - 0.9, -1.25)
    t_hi = max(float(key_t.max()) + 0.9, 1.25)
    context = (
        (np.abs(qn) <= 0.72)
        & (qt >= t_lo)
        & (qt <= t_hi)
        & (np.abs(qd) <= 0.85)
    )
    candidate = voxel_mask(world, region["voxel_keys"], root_voxel_size)
    layer_a = candidate & _frame_mask(pose_ids, region["layer_a_keyframes"])
    layer_b = candidate & _frame_mask(pose_ids, region["layer_b_keyframes"])

    context_idx = _sample(context & ~layer_a & ~layer_b, 13000)
    a_idx = _sample(layer_a, 5500)
    b_idx = _sample(layer_b, 5500)
    ax.scatter(qn[context_idx], qt[context_idx], s=0.28, c=NEUTRAL, alpha=0.18,
               linewidths=0, rasterized=True)
    ax.scatter(qn[a_idx], qt[a_idx], s=1.1, c=BLUE, alpha=0.72,
               linewidths=0, rasterized=True)
    ax.scatter(qn[b_idx], qt[b_idx], s=1.1, c=ORANGE, alpha=0.72,
               linewidths=0, rasterized=True)

    means = []
    for indices, color in ((a_idx, BLUE), (b_idx, ORANGE)):
        if len(indices):
            value = float(np.median(qn[indices]))
            means.append(value)
            ax.axvline(value, color=color, lw=0.8, alpha=0.9, zorder=5)
    if len(means) == 2:
        y = t_hi - 0.12 * (t_hi - t_lo)
        ax.annotate("", (means[0], y), (means[1], y),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.65))

    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(t_lo, t_hi)
    # Keep the physical units explicit on both axes, but do not force equal
    # display aspect: a 10 cm double trace next to a 10 m surface would become
    # an unreadably thin strip.  The normal-axis scale and layer-center lines
    # preserve the quantitative separation without hiding the evidence.
    ax.set_xlabel("surface-normal offset (m)", labelpad=1.5)
    ax.set_ylabel("along surface (m)", labelpad=1.5)
    ax.grid(color="#e8ebef", lw=0.45, zorder=-10)
    for spine in ax.spines.values():
        spine.set_color("#79838e")
    ax.tick_params(labelsize=6, length=2.2, pad=1.5, colors="#47515c")
    ax.set_title(
        f"#{rank} · {region['separation_typical_m'] * 100:.1f} cm separation",
        loc="left", fontsize=7.7, fontweight="bold", color=INK, pad=3,
    )
    ax.text(
        0.99, 0.99,
        f"{region['voxel_count']} patches · {region['point_count']:,} pts\n"
        f"overlap {region['layer_overlap_ratio']:.2f}\n"
        f"kf {region['layer_a_keyframes'][0]}–{region['layer_a_keyframes'][1]} / "
        f"{region['layer_b_keyframes'][0]}–{region['layer_b_keyframes'][1]}",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color="#39434d",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82),
    )



def _draw_context(
    ax: plt.Axes,
    region: dict,
    world: np.ndarray,
    pose_ids: np.ndarray,
    root_voxel_size: float,
) -> None:
    """Show the actual surrounding structure at a readable scale."""
    center = np.asarray(region.get("region_centroid_xyz", region["center_xyz"]), dtype=float)
    normal = _canonical(np.asarray(region["normal"], dtype=float))
    radius = 5.0
    horizontal_surface = abs(float(normal[2])) > 0.75
    if horizontal_surface:
        key_centers = (
            np.asarray(region["voxel_keys"], dtype=float) + 0.5
        ) * root_voxel_size
        key_xy = key_centers[:, :2] - key_centers[:, :2].mean(axis=0)
        if len(key_xy) > 1 and float(np.linalg.norm(key_xy)) > 0.2:
            _, _, vh = np.linalg.svd(key_xy, full_matrices=False)
            tangent_xy = vh[0]
        else:
            tangent_xy = np.array([1.0, 0.0])
        depth_xy = np.array([-tangent_xy[1], tangent_xy[0]])
        relative_xy = world[:, :2] - center[:2]
        display_x = relative_xy @ tangent_xy
        display_y = world[:, 2] - center[2]
        nearby = (
            (np.abs(display_x) <= radius)
            & (np.abs(relative_xy @ depth_xy) <= 1.5)
            & (np.abs(display_y) <= 3.0)
        )
        xlabel, ylabel = "along surface (m)", "z offset (m)"
        title = "real structural context (side view)"
    else:
        display_x, display_y = world[:, 0], world[:, 1]
        nearby = (
            (np.abs(world[:, 0] - center[0]) <= radius)
            & (np.abs(world[:, 1] - center[1]) <= radius)
            & (np.abs(world[:, 2] - center[2]) <= 1.5)
        )
        xlabel, ylabel = "x (m)", "y (m)"
        title = "real structural context (top view)"
    layer_a = nearby & _frame_mask(pose_ids, region["layer_a_keyframes"])
    layer_b = nearby & _frame_mask(pose_ids, region["layer_b_keyframes"])
    neutral_idx = _sample(nearby & ~layer_a & ~layer_b, 22000)
    a_idx = _sample(layer_a, 7500)
    b_idx = _sample(layer_b, 7500)
    ax.scatter(display_x[neutral_idx], display_y[neutral_idx], s=0.22, c=NEUTRAL,
               alpha=0.18, linewidths=0, rasterized=True)
    ax.scatter(display_x[a_idx], display_y[a_idx], s=0.45, c=BLUE,
               alpha=0.33, linewidths=0, rasterized=True)
    ax.scatter(display_x[b_idx], display_y[b_idx], s=0.45, c=ORANGE,
               alpha=0.33, linewidths=0, rasterized=True)
    # Show the exact audited root voxels without hiding their points.
    for key in region["voxel_keys"]:
        origin = np.asarray(key, dtype=float) * root_voxel_size
        if horizontal_surface:
            key_center = origin + 0.5 * root_voxel_size
            box_x = (
                float((key_center[:2] - center[:2]) @ tangent_xy)
                - 0.5 * root_voxel_size
            )
            box_y = float(key_center[2] - center[2]) - 0.5 * root_voxel_size
        else:
            box_x, box_y = float(origin[0]), float(origin[1])
        rect = mpl.patches.Rectangle(
            (box_x, box_y), root_voxel_size, root_voxel_size,
            fill=False, ec="#c33d36", lw=0.75, linestyle=(0, (2, 1.5)), zorder=8,
        )
        ax.add_patch(rect)
    marker_x, marker_y = ((0.0, 0.0) if horizontal_surface else (center[0], center[1]))
    ax.plot(marker_x, marker_y, marker="+", ms=5, mew=0.9, color="#c33d36", zorder=9)
    if horizontal_surface:
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-3.0, 3.0)
    else:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_title(title, loc="left", fontsize=6.8, color="#47515c", pad=2)
    ax.tick_params(labelsize=5.5, length=2, pad=1.2, colors="#59636e")
    ax.set_facecolor("#fafbfc")
    for spine in ax.spines.values():
        spine.set_color("#79838e")
        spine.set_linewidth(0.55)


def _render_sheet(
    regions: list[dict], world: np.ndarray, pose_ids: np.ndarray,
    start_rank: int, output_stem: Path, dataset_label: str, round_label: str,
    root_voxel_size: float,
) -> None:
    rows, cols = len(regions), 1
    fig = plt.figure(figsize=(7.15, 2.65 * rows + 0.85))
    outer = fig.add_gridspec(rows, cols, left=0.055, right=0.985, bottom=0.105,
                             top=0.885, wspace=0.20, hspace=0.42)
    fig.suptitle(f"{dataset_label} {round_label} · loop-explainable ghost audit",
                 x=0.055, y=0.968, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.936,
        "Gray: nearby structure   traversal A: blue   traversal B: orange   "
        "Dashed red boxes: exact detected surface patches.",
        ha="left", va="center", fontsize=7.5, color="#4a5561",
    )
    for offset in range(rows * cols):
        if offset >= len(regions):
            continue
        nested = outer[offset // cols, offset % cols].subgridspec(
            1, 2, width_ratios=(1.08, 0.92), wspace=0.22
        )
        ax_context = fig.add_subplot(nested[0, 0])
        ax_section = fig.add_subplot(nested[0, 1])
        _draw_context(
            ax_context, regions[offset], world, pose_ids, root_voxel_size
        )
        _draw_section(
            ax_section, start_rank + offset, regions[offset], world, pose_ids,
            root_voxel_size,
        )
    fig.text(
        0.5, 0.034,
        "Uniform view; no candidate or viewpoint was manually selected.\n"
        "Left: structural context. Right: normal section; a visible ghost forms two parallel traces.",
        ha="center", va="center", fontsize=6.2, color="#5b6570",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tum", type=Path)
    parser.add_argument("report", type=Path,
                        help="BALM report JSON or legacy region-list JSON")
    parser.add_argument("session", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--balm-report", type=Path,
                        help="Run report supplying the true root voxel size")
    parser.add_argument("--root-voxel-size", type=float)
    parser.add_argument("--dataset-label")
    parser.add_argument("--round-label", default="R1")
    args = parser.parse_args()
    tum = args.tum.expanduser()
    report_path = args.report.expanduser()
    session = args.session.expanduser()
    output_dir = args.output_dir.expanduser()
    payload = json.loads(report_path.read_text())
    regions = (
        payload.get("ghost_regions_all", payload.get("ghost_regions", []))
        if isinstance(payload, dict) else payload
    )
    voxel_report = args.balm_report or (
        report_path if isinstance(payload, dict) and "params" in payload else None
    )
    root_voxel_size = load_balm_root_voxel_size(
        voxel_report, args.root_voxel_size
    )
    dataset_label = args.dataset_label or session.name
    max_observation_range = (
        float(payload.get("params", {}).get("max_observation_range", MAX_OBSERVATION_RANGE))
        if isinstance(payload, dict) else MAX_OBSERVATION_RANGE
    )
    report_params = payload.get("params", {}) if isinstance(payload, dict) else {}
    preparation_params = BalmParams(
        root_voxel_size=root_voxel_size,
        downsample_leaf=float(report_params.get("downsample_leaf", 0.2)),
        max_range=float(report_params.get("max_range", 80.0)),
    )
    trajectory = load_tum_trajectory(tum)
    workspace = RegistrationWorkspace(session / "key_point_frame", trajectory)
    rotations = trajectory.transforms_world_sensor[:, :3, :3]
    translations = trajectory.transforms_world_sensor[:, :3, 3]

    world_parts: list[np.ndarray] = []
    pose_parts: list[np.ndarray] = []
    for index in range(trajectory.size):
        local = prepare_local_cloud(
            workspace.load_local_points_uncached(int(index)), preparation_params
        ).astype(np.float64)
        world = local @ rotations[index].T + translations[index]
        near = np.einsum("ij,ij->i", world - translations[index],
                         world - translations[index]) <= max_observation_range**2
        world_parts.append(world[near].astype(np.float32))
        pose_parts.append(np.full(int(near.sum()), index, dtype=np.int32))
        if (index + 1) % 500 == 0:
            print(f"  loaded {index + 1}/{trajectory.size}", flush=True)
    world = np.vstack(world_parts)
    pose_ids = np.concatenate(pose_parts)

    page_count = math.ceil(len(regions) / 6)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = {
        "dataset": dataset_label,
        "round": args.round_label,
        "root_voxel_size_m": root_voxel_size,
        "candidate_count": len(regions),
        "candidates": [
            {
                "rank": rank,
                "center_xyz": region.get("region_centroid_xyz", region["center_xyz"]),
                "normal": region["normal"],
                "separation_typical_m": region["separation_typical_m"],
                "point_count": region["point_count"],
                "voxel_count": region["voxel_count"],
                "layer_overlap_ratio": region["layer_overlap_ratio"],
                "layer_a_keyframes": region["layer_a_keyframes"],
                "layer_b_keyframes": region["layer_b_keyframes"],
            }
            for rank, region in enumerate(regions, 1)
        ],
    }
    slug = "".join(c if c.isalnum() else "_" for c in dataset_label).strip("_")
    (output_dir / f"{slug}_{args.round_label}_audit_source_data.json").write_text(
        json.dumps(source_data, indent=2) + "\n", encoding="utf-8"
    )
    for page in range(page_count):
        begin, end = page * 6, min((page + 1) * 6, len(regions))
        stem = output_dir / (
            f"{slug}_{args.round_label}_refined_ghost_human_audit_page{page + 1}"
        )
        _render_sheet(
            regions[begin:end], world, pose_ids, begin + 1, stem,
            dataset_label, args.round_label, root_voxel_size,
        )
        print(f"  rendered ranks {begin + 1}-{end} -> {stem}.png", flush=True)


if __name__ == "__main__":
    main()
