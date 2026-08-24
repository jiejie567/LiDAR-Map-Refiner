#!/usr/bin/env python3
"""Track selected R1 ghost surfaces through R1--R4 without re-detection.

The R1 point identities, same-side observations, per-voxel two-layer labels,
surface normals and plotting basis are frozen.  Later panels only reproject
those observations with each round's optimized poses.  This prevents a later
detector run from silently substituting a different voxel or structure.

Usage:
  make_refined_ghost_round_progression.py \
    --tums R1.tum R2.tum R3.tum R4.tum --report regions.json \
    --keyframes DIR --map-pcd R1.pcd --out-dir DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / "gui"))
sys.path.insert(0, str(TOOL / "gui/manual_loop_closure/python_optimizer"))
from manual_loop_closure import RegistrationWorkspace, load_tum_trajectory  # noqa: E402
from balm import (  # noqa: E402
    BalmParams,
    _canonical_plane_normal,
    _two_means_1d,
    prepare_local_cloud,
)
from merge_pcds import read_pcd  # noqa: E402
from experiment_io import (  # noqa: E402
    load_balm_root_voxel_size,
    validate_round_inputs,
)


BLUE = "#2474b7"
ORANGE = "#e1812c"
GRAY = "#8d98a4"
PALE = "#dfe4e9"
INK = "#17212b"
ALERT = "#b74d43"
THRESHOLD_M = 0.08
MAX_RANGE_M = 20.0

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def _key_tuple(key: np.ndarray | list[int]) -> tuple[int, int, int]:
    return tuple(int(value) for value in key)


def _interval_fraction(ids: np.ndarray, interval: list[int]) -> float:
    if not len(ids):
        return 0.0
    return float(np.mean((ids >= interval[0]) & (ids <= interval[1])))


def _view_basis(region: dict, root_voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    normal = _canonical_plane_normal(np.asarray(region["normal"], dtype=float))
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    keys = (
        np.asarray(region["voxel_keys"], dtype=float) + 0.5
    ) * root_voxel_size
    centered = keys - keys.mean(axis=0)
    in_plane = centered - np.outer(centered @ normal, normal)
    if len(keys) > 1 and float(np.linalg.norm(in_plane)) > 0.25:
        _, _, vh = np.linalg.svd(in_plane, full_matrices=False)
        tangent = vh[0] - normal * float(vh[0] @ normal)
    else:
        tangent = np.array([0.0, 0.0, 1.0])
        tangent -= normal * float(tangent @ normal)
        if np.linalg.norm(tangent) < 0.25:
            tangent = np.array([1.0, 0.0, 0.0])
            tangent -= normal * float(tangent @ normal)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    return normal, tangent


def _select_r1_layers(
    region: dict,
    voxel_key: tuple[int, int, int],
    local: np.ndarray,
    pose_ids: np.ndarray,
    r1_poses: np.ndarray,
    component_voxel: dict | None = None,
) -> dict:
    """Reproduce the detector's same-side split for one component voxel."""
    transforms = r1_poses[pose_ids]
    world = (
        np.einsum("nij,nj->ni", transforms[:, :3, :3], local)
        + transforms[:, :3, 3]
    )
    sensors = transforms[:, :3, 3]
    mean = world.mean(axis=0)
    centered = world - mean
    if component_voxel is None:
        _, eigenvectors = np.linalg.eigh(centered.T @ centered / len(world))
        normal = _canonical_plane_normal(eigenvectors[:, 0])
        expected_side = None
        expected_separation = float(region["separation_typical_m"])
    else:
        normal = np.asarray(component_voxel["normal"], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        expected_side = int(component_voxel["view_side"])
        expected_separation = float(component_voxel["separation_m"])
    view_dot = np.einsum("ij,j->i", sensors - world, normal)
    range_a = region["layer_a_keyframes"]
    range_b = region["layer_b_keyframes"]
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    side_options = (
        ((view_dot >= 0.0),) if expected_side == 1
        else ((view_dot < 0.0),) if expected_side == -1
        else (view_dot >= 0.0, view_dot < 0.0)
    )
    for side_mask in side_options:
        if int(side_mask.sum()) < 40:
            continue
        side_indices = np.flatnonzero(side_mask)
        offsets = (world[side_indices] - mean) @ normal
        split = _two_means_1d(offsets)
        if split is None:
            continue
        low_center, high_center, low_mask = split
        low_ids = pose_ids[side_indices[low_mask]]
        high_ids = pose_ids[side_indices[~low_mask]]
        direct = min(
            _interval_fraction(low_ids, range_a),
            _interval_fraction(high_ids, range_b),
        )
        crossed = min(
            _interval_fraction(low_ids, range_b),
            _interval_fraction(high_ids, range_a),
        )
        traversal_score = max(direct, crossed)
        separation = float(high_center - low_center)
        separation_error = abs(separation - expected_separation)
        score = traversal_score - 0.25 * separation_error
        candidates.append(
            (score, side_indices, low_mask, normal.copy(), separation)
        )
    if not candidates:
        raise RuntimeError(f"No same-side R1 layer split for voxel {voxel_key}")
    _, selected, layer_low, normal, separation = max(candidates, key=lambda item: item[0])
    return {
        "local": local[selected],
        "pose_ids": pose_ids[selected],
        "layer_low": layer_low,
        "normal": normal,
        "r1_separation_m": separation,
    }


def _sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def _render_page(
    page_regions: list[dict],
    page_tracks: list[dict],
    map_xyz: np.ndarray,
    round_labels: list[str],
    loop_counts: list[int],
    output_stem: Path,
    dataset_label: str,
    root_voxel_size: float,
) -> None:
    rows = len(page_regions)
    round_count = len(round_labels)
    single_case = rows == 1
    fig_height = 4.65 if single_case else 3.05 * rows + 0.75
    fig = plt.figure(figsize=(3.4 + 1.62 * round_count, fig_height))
    grid = fig.add_gridspec(
        rows,
        round_count + 2,
        left=0.045,
        right=0.985,
        bottom=0.16 if single_case else 0.075,
        top=0.77 if single_case else 0.89,
        width_ratios=(1.28, *([1.0] * round_count), 1.18),
        wspace=0.25,
        hspace=0.38,
    )
    fig.suptitle(
        f"{dataset_label} · fixed R1 ghost evolution",
        x=0.045, y=0.965, ha="left",
        fontsize=10.5 if single_case else 12.5,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.045, 0.91 if single_case else 0.932,
        "The same R1 observations and layer labels are reprojected in every round; "
        "blue/orange never switch identity.",
        ha="left", fontsize=6.6 if single_case else 7.2, color="#53606c",
    )

    for row, (region, track) in enumerate(zip(page_regions, page_tracks)):
        center = track["reference_center"]
        ax_context = fig.add_subplot(grid[row, 0])
        context = (
            (np.abs(map_xyz[:, 0] - center[0]) <= 4.5)
            & (np.abs(map_xyz[:, 1] - center[1]) <= 4.5)
            & (np.abs(map_xyz[:, 2] - center[2]) <= 1.6)
        )
        context_points = map_xyz[context]
        ax_context.scatter(
            context_points[:, 0], context_points[:, 1], s=0.22, c=GRAY,
            alpha=0.22, linewidths=0, rasterized=True,
        )
        r1_world = track["rounds"][0]["world"]
        layer = track["layer_low"]
        ax_context.scatter(r1_world[layer, 0], r1_world[layer, 1], s=0.65,
                           c=BLUE, alpha=0.42, linewidths=0, rasterized=True)
        ax_context.scatter(r1_world[~layer, 0], r1_world[~layer, 1], s=0.65,
                           c=ORANGE, alpha=0.42, linewidths=0, rasterized=True)
        for key in region["voxel_keys"]:
            ax_context.add_patch(
                mpl.patches.Rectangle(
                    np.asarray(key[:2], dtype=float) * root_voxel_size,
                    root_voxel_size, root_voxel_size,
                    fill=False, ec=ALERT, lw=0.7, linestyle=(0, (2, 1.5)),
                )
            )
        ax_context.set_xlim(center[0] - 4.5, center[0] + 4.5)
        ax_context.set_ylim(center[1] - 4.5, center[1] + 4.5)
        ax_context.set_aspect("equal", adjustable="box")
        ax_context.set_title(
            f"#{track['rank']} · fixed structure",
            loc="left", fontsize=7.6, fontweight="bold", color=INK,
        )
        ax_context.set_xlabel("x (m)")
        ax_context.set_ylabel("y (m)")
        ax_context.tick_params(labelsize=5.6, length=2, pad=1)

        common_xlim = track["normal_xlim"]
        common_ylim = track["tangent_ylim"]
        for column, item in enumerate(track["rounds"], start=1):
            ax = fig.add_subplot(grid[row, column])
            layer = track["layer_low"]
            low_idx = _sample(np.flatnonzero(layer), 4500)
            high_idx = _sample(np.flatnonzero(~layer), 4500)
            ax.scatter(
                item["normal_offset"][low_idx], item["tangent_offset"][low_idx],
                s=0.65, c=BLUE, alpha=0.42, linewidths=0, rasterized=True,
            )
            ax.scatter(
                item["normal_offset"][high_idx], item["tangent_offset"][high_idx],
                s=0.65, c=ORANGE, alpha=0.42, linewidths=0, rasterized=True,
            )
            for value, color in zip(item["layer_centers"], (BLUE, ORANGE)):
                ax.axvline(value, c=color, lw=0.9)
            ax.set_xlim(*common_xlim)
            ax.set_ylim(*common_ylim)
            ax.grid(c=PALE, lw=0.4, zorder=-5)
            ax.set_title(
                f"{item['label']} · {loop_counts[column - 1]} loops\n"
                f"{item['separation_m'] * 100:.1f} cm",
                fontsize=7.2, fontweight="bold", pad=3,
            )
            ax.set_xlabel("normal offset (m)")
            if column == 1:
                ax.set_ylabel("along surface (m)")
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=5.6, length=2, pad=1)

        ax_trend = fig.add_subplot(grid[row, round_count + 1])
        values_cm = np.asarray(
            [item["separation_m"] * 100.0 for item in track["rounds"]]
        )
        xx = np.arange(round_count)
        ax_trend.axhspan(0.0, 8.0, color=PALE, alpha=0.55, lw=0)
        ax_trend.axhline(8.0, color=GRAY, lw=0.75, ls=(0, (3, 2)))
        ax_trend.plot(xx, values_cm, color=ALERT, lw=1.2, marker="o", ms=3.3)
        pad = max(0.55, float(values_cm.max()) * 0.035)
        for xpos, value in zip(xx, values_cm):
            ax_trend.text(xpos, value + pad, f"{value:.1f}", ha="center",
                          va="bottom", fontsize=6.1, fontweight="bold", color=ALERT)
        ymax = max(10.0, float(values_cm.max()) * 1.28)
        ax_trend.set_ylim(0.0, ymax)
        ax_trend.set_xlim(-0.3, max(0.3, round_count - 0.7))
        ax_trend.set_xticks(xx, round_labels)
        ax_trend.set_ylabel("separation (cm)")
        ax_trend.grid(axis="y", color=PALE, lw=0.4)
        reduction = 100.0 * (values_cm[0] - values_cm[-1]) / values_cm[0]
        outcome = "closed" if values_cm[-1] < 8.0 else "persistent"
        if reduction >= 0.0:
            change_text = f"{reduction:.0f}% reduction"
        else:
            change_text = f"{-reduction:.0f}% increase"
        ax_trend.set_title(
            f"{outcome} · {change_text}",
            loc="left", fontsize=7.1, fontweight="bold", color=INK,
        )
        ax_trend.tick_params(labelsize=5.8, length=2, pad=1)

    fig.text(
        0.985, 0.045 if single_case else 0.018,
        "Shaded band: <8 cm detector threshold. Distances are weighted across the "
        "same component voxels using fixed R1 normals and layer IDs.",
        ha="right", fontsize=5.8, color="#58636e",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tums", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True,
                        help="R1 BALM report JSON or legacy region-list JSON")
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--map-pcd", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--round-labels", nargs="+")
    parser.add_argument("--loop-counts", type=int, nargs="+")
    parser.add_argument("--balm-report", type=Path)
    parser.add_argument("--root-voxel-size", type=float)
    parser.add_argument("--dataset-label")
    args = parser.parse_args()

    if args.round_labels is None:
        args.round_labels = [f"R{index + 1}" for index in range(len(args.tums))]
    if args.loop_counts is None:
        args.loop_counts = [0] * len(args.tums)
    validate_round_inputs(args.tums, args.round_labels, args.loop_counts)

    trajectories = [load_tum_trajectory(path) for path in args.tums]
    if len({trajectory.size for trajectory in trajectories}) != 1:
        raise RuntimeError("Round trajectories have different pose counts")
    payload = json.loads(args.report.read_text())
    all_regions = (
        payload.get("ghost_regions_all", payload.get("ghost_regions", []))
        if isinstance(payload, dict) else payload
    )
    if args.ranks is None:
        args.ranks = list(range(1, len(all_regions) + 1))
    voxel_report = args.balm_report or (
        args.report if isinstance(payload, dict) and "params" in payload else None
    )
    root_voxel_size = load_balm_root_voxel_size(
        voxel_report, args.root_voxel_size
    )
    report_params = payload.get("params", {}) if isinstance(payload, dict) else {}
    preparation_params = BalmParams(
        root_voxel_size=root_voxel_size,
        downsample_leaf=float(report_params.get("downsample_leaf", 0.2)),
        max_range=float(report_params.get("max_range", 80.0)),
    )
    max_observation_range = float(
        report_params.get("max_observation_range", MAX_RANGE_M)
    )
    dataset_label = args.dataset_label or args.keyframes.parent.name
    regions = [all_regions[rank - 1] for rank in args.ranks]
    selected_keys = {
        _key_tuple(key)
        for region in regions
        for key in region["voxel_keys"]
    }
    workspace = RegistrationWorkspace(args.keyframes, trajectories[0])
    r1_poses = trajectories[0].transforms_world_sensor
    local_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    pose_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)

    for frame_index in range(trajectories[0].size):
        local = prepare_local_cloud(
            workspace.load_local_points_uncached(frame_index), preparation_params
        ).astype(np.float64)
        transform = r1_poses[frame_index]
        world = local @ transform[:3, :3].T + transform[:3, 3]
        near = np.einsum("ij,ij->i", world - transform[:3, 3],
                         world - transform[:3, 3]) <= max_observation_range**2
        local = local[near]
        keys = np.floor(world[near] / root_voxel_size).astype(np.int64)
        for key in selected_keys:
            keep = np.all(keys == np.asarray(key, dtype=np.int64), axis=1)
            if np.any(keep):
                local_by_key[key].append(local[keep])
                pose_by_key[key].append(np.full(int(keep.sum()), frame_index, dtype=np.int32))
        if (frame_index + 1) % 500 == 0:
            print(f"  scanned {frame_index + 1}/{trajectories[0].size}", flush=True)

    tracks: list[dict] = []
    csv_rows: list[dict] = []
    for rank, region in zip(args.ranks, regions):
        component_by_key = {
            _key_tuple(item["voxel_key"]): item
            for item in region.get("component_voxels", [])
        }
        voxel_groups: list[dict] = []
        for key_value in region["voxel_keys"]:
            key = _key_tuple(key_value)
            if key not in local_by_key:
                raise RuntimeError(f"No R1 observations for region #{rank}, voxel {key}")
            group = _select_r1_layers(
                region,
                key,
                np.vstack(local_by_key[key]),
                np.concatenate(pose_by_key[key]),
                r1_poses,
                component_by_key.get(key),
            )
            group["voxel_key"] = key
            voxel_groups.append(group)

        local = np.vstack([group["local"] for group in voxel_groups])
        pose_ids = np.concatenate([group["pose_ids"] for group in voxel_groups])
        layer_low = np.concatenate([group["layer_low"] for group in voxel_groups])
        voxel_ids = np.concatenate([
            np.full(len(group["local"]), index, dtype=np.int32)
            for index, group in enumerate(voxel_groups)
        ])
        reference_normal, tangent = _view_basis(region, root_voxel_size)
        r1_transforms = r1_poses[pose_ids]
        r1_world = (
            np.einsum("nij,nj->ni", r1_transforms[:, :3, :3], local)
            + r1_transforms[:, :3, 3]
        )
        reference_center = r1_world.mean(axis=0)
        round_items: list[dict] = []
        for label, trajectory in zip(args.round_labels, trajectories):
            transforms = trajectory.transforms_world_sensor[pose_ids]
            world = (
                np.einsum("nij,nj->ni", transforms[:, :3, :3], local)
                + transforms[:, :3, 3]
            )
            normal_offset = (world - reference_center) @ reference_normal
            tangent_offset = (world - reference_center) @ tangent
            component_separations = []
            component_weights = []
            for voxel_index, group in enumerate(voxel_groups):
                selected = voxel_ids == voxel_index
                group_layer = layer_low[selected]
                group_offset = (world[selected] - reference_center) @ group["normal"]
                component_separations.append(
                    abs(float(group_offset[group_layer].mean())
                        - float(group_offset[~group_layer].mean()))
                )
                component_weights.append(int(selected.sum()))
            separation = float(np.average(component_separations, weights=component_weights))
            low_center = float(normal_offset[layer_low].mean())
            high_center = float(normal_offset[~layer_low].mean())
            midpoint = (low_center + high_center) / 2.0
            round_items.append(
                {
                    "label": label,
                    "world": world,
                    "normal_offset": normal_offset - midpoint,
                    "tangent_offset": tangent_offset,
                    "layer_centers": [low_center - midpoint, high_center - midpoint],
                    "separation_m": separation,
                }
            )
            csv_rows.append(
                {
                    "candidate_rank": rank,
                    "round": label,
                    "tracked_point_count": len(local),
                    "tracked_keyframe_count": len(np.unique(pose_ids)),
                    "component_voxel_count": len(voxel_groups),
                    "separation_m": separation,
                    "below_8cm": separation < THRESHOLD_M,
                }
            )
        all_normal = np.concatenate([item["normal_offset"] for item in round_items])
        all_tangent = np.concatenate([item["tangent_offset"] for item in round_items])
        normal_limit = max(0.22, float(np.percentile(np.abs(all_normal), 99.5)) * 1.12)
        tangent_low, tangent_high = np.percentile(all_tangent, [0.5, 99.5])
        tangent_pad = max(0.15, 0.08 * float(tangent_high - tangent_low))
        tracks.append(
            {
                "rank": rank,
                "reference_center": reference_center,
                "layer_low": layer_low,
                "normal_xlim": (-normal_limit, normal_limit),
                "tangent_ylim": (float(tangent_low - tangent_pad), float(tangent_high + tangent_pad)),
                "rounds": round_items,
                "tracked_point_count": len(local),
                "tracked_keyframe_count": len(np.unique(pose_ids)),
            }
        )
        report_separation = float(region["separation_typical_m"])
        r1_error = abs(round_items[0]["separation_m"] - report_separation)
        if r1_error > 0.01:
            raise RuntimeError(
                f"Candidate #{rank}: fixed R1 separation "
                f"{round_items[0]['separation_m']:.3f} m differs from detector "
                f"typical {report_separation:.3f} m by {r1_error:.3f} m"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in dataset_label).strip("_")
    with (args.out_dir / f"{slug}_R1_selected_ghost_round_progression.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    source_summary = {
        "tracking_rule": (
            "R1 point identity, observing side, per-voxel two-means layer ID, "
            "surface normal and plotting basis are fixed; later rounds change poses only."
        ),
        "selected_ranks": list(args.ranks),
        "round_labels": list(args.round_labels),
        "active_loop_counts": list(args.loop_counts),
        "detection_threshold_m": THRESHOLD_M,
        "root_voxel_size_m": root_voxel_size,
        "candidates": [
            {
                "rank": track["rank"],
                "tracked_point_count": track["tracked_point_count"],
                "tracked_keyframe_count": track["tracked_keyframe_count"],
                "separation_m": [round(item["separation_m"], 6) for item in track["rounds"]],
                "final_below_8cm": track["rounds"][-1]["separation_m"] < THRESHOLD_M,
            }
            for track in tracks
        ],
    }
    (args.out_dir / f"{slug}_R1_selected_ghost_round_progression_source_data.json").write_text(
        json.dumps(source_summary, indent=2) + "\n", encoding="utf-8"
    )
    # Exact immutable R1 observations and labels make every later projection
    # independently reproducible without relying on candidate re-detection.
    for track, region in zip(tracks, regions):
        component_by_key = {
            _key_tuple(item["voxel_key"]): item
            for item in region.get("component_voxels", [])
        }
        groups = []
        for key_value in region["voxel_keys"]:
            key = _key_tuple(key_value)
            group = _select_r1_layers(
                region, key, np.vstack(local_by_key[key]),
                np.concatenate(pose_by_key[key]), r1_poses,
                component_by_key.get(key),
            )
            groups.append(group)
        np.savez_compressed(
            args.out_dir / f"{slug}_R1_candidate{track['rank']:02d}_fixed_points.npz",
            local_xyz=np.vstack([group["local"] for group in groups]),
            pose_ids=np.concatenate([group["pose_ids"] for group in groups]),
            layer_low=np.concatenate([group["layer_low"] for group in groups]),
            voxel_ids=np.concatenate([
                np.full(len(group["local"]), index, dtype=np.int32)
                for index, group in enumerate(groups)
            ]),
            voxel_normals=np.vstack([group["normal"] for group in groups]),
            voxel_keys=np.asarray(region["voxel_keys"], dtype=np.int64),
            root_voxel_size_m=np.asarray(root_voxel_size),
        )

    map_cloud = read_pcd(args.map_pcd)
    map_xyz = np.column_stack(
        [map_cloud.data[name] for name in ("x", "y", "z")]
    ).astype(float)
    pages = math.ceil(len(regions) / 3)
    for page in range(pages):
        begin, end = page * 3, min((page + 1) * 3, len(regions))
        stem = args.out_dir / f"{slug}_R1_selected_ghost_round_progression_page{page + 1}"
        _render_page(
            regions[begin:end], tracks[begin:end], map_xyz,
            list(args.round_labels), list(args.loop_counts), stem,
            dataset_label, root_voxel_size,
        )
        print(f"  rendered candidates {args.ranks[begin:end]} -> {stem}.png", flush=True)
    print(json.dumps(source_summary, indent=2))


if __name__ == "__main__":
    main()
