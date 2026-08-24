#!/usr/bin/env python3
"""
Visualize a pose-graph stored in g2o format.

Usage examples
--------------
# Render interactively
python3 visualize_pose_graph.py --g2o pose_graph.g2o --show

# Save a figure (no interactive window)
python3 visualize_pose_graph.py --g2o pose_graph.g2o --output pose_graph.png

Features
--------
- Supports SE2 / SE3 vertices.
- Colors odometry edges (difference in index below a threshold) and loop closures.
- Optional 3D projection.
- Can annotate node IDs and display basic statistics.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

VertexDict = Dict[int, Tuple[float, float, float]]
EdgeList = List[Tuple[int, int, str]]
GnssTypeMap = Dict[int, str]

# Anchor vertex ID used when exporting pose priors to standard g2o format.
ANCHOR_VERTEX_ID = (2 ** 31) - 2  # matches std::numeric_limits<int>::max() - 1


def parse_g2o(path: Path) -> Tuple[VertexDict, EdgeList, Set[int], GnssTypeMap]:
    """Parse vertices, edges, prior nodes, and GNSS-constrained nodes from a g2o file."""
    vertices: VertexDict = {}
    edges: EdgeList = []
    prior_nodes: Set[int] = set()
    gnss_nodes: GnssTypeMap = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                tokens = line.split()
                if len(tokens) >= 4 and tokens[1] == "GNSS_PRIOR":
                    try:
                        constraint_type = tokens[2]
                        node_id = int(tokens[3])
                        gnss_nodes[node_id] = constraint_type
                    except ValueError:
                        pass
                continue

            tokens = line.split()
            tag = tokens[0]

            if tag.startswith("VERTEX"):
                idx = int(tokens[1])
                if idx == ANCHOR_VERTEX_ID:
                    # Skip visualization of artificial anchor vertex used for priors.
                    continue
                if tag == "VERTEX_SE3:QUAT":
                    x, y, z = map(float, tokens[2:5])
                elif tag in {"VERTEX_SE2", "VERTEX_XY"}:
                    x, y = map(float, tokens[2:4])
                    z = 0.0
                else:
                    continue  # unhandled type
                vertices[idx] = (x, y, z)

            elif tag.startswith("EDGE"):
                if len(tokens) < 3:
                    continue
                i, j = int(tokens[1]), int(tokens[2])
                if i == ANCHOR_VERTEX_ID or j == ANCHOR_VERTEX_ID:
                    prior_nodes.add(j if i == ANCHOR_VERTEX_ID else i)
                    continue
                edges.append((i, j, tag))

    return vertices, edges, prior_nodes, gnss_nodes


def classify_edges(edges: EdgeList, odom_threshold: int) -> Tuple[EdgeList, EdgeList]:
    """Split edges into odometry and loop closures."""
    odom_edges: EdgeList = []
    loop_edges: EdgeList = []

    for i, j, tag in edges:
        if abs(i - j) <= odom_threshold:
            odom_edges.append((i, j, tag))
        else:
            loop_edges.append((i, j, tag))
    return odom_edges, loop_edges


def to_numpy(vertices: VertexDict) -> Tuple[np.ndarray, List[int]]:
    """Convert dict of vertices into ordered arrays."""
    if not vertices:
        raise ValueError("No vertices parsed from g2o file.")
    ids = sorted(vertices.keys())
    xyz = np.array([vertices[idx] for idx in ids])
    return xyz, ids


def _edge_segments(edges: EdgeList, coords: np.ndarray, id_to_index: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    segments = []
    magnitudes = []
    for i, j, _ in edges:
        if i not in id_to_index or j not in id_to_index:
            continue
        pi, pj = coords[id_to_index[i]], coords[id_to_index[j]]
        segments.append(np.vstack([pi, pj]))
        magnitudes.append(np.linalg.norm(pj - pi))
    if not segments:
        return np.zeros((0, 2, coords.shape[1])), np.zeros(0)
    return np.stack(segments), np.array(magnitudes)


def plot_pose_graph(vertices: VertexDict,
                    odom_edges: EdgeList,
                    loop_edges: EdgeList,
                    prior_nodes: Set[int],
                    gnss_nodes: GnssTypeMap,
                    args: argparse.Namespace) -> None:
    coords, ordered_ids = to_numpy(vertices)
    id_to_index = {idx: k for k, idx in enumerate(ordered_ids)}
    first_node_id = ordered_ids[0]

    if args.three_d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")

    node_kwargs = dict(s=20, c="#555555", alpha=0.5)

    if args.three_d:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **node_kwargs)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], **node_kwargs)

    gnss_color_map = {
        "XYZ": "#2ca02c",
        "POSE": "#17becf",
        "XY": "#8c564b",
    }
    gnss_groups: Dict[str, List[int]] = {}
    for node_id, constraint_type in gnss_nodes.items():
        if node_id not in id_to_index:
            continue
        gnss_groups.setdefault(constraint_type, []).append(id_to_index[node_id])

    for constraint_type, indices in gnss_groups.items():
        color = gnss_color_map.get(constraint_type, "#2ca02c")
        label = f"GNSS {constraint_type} ({len(indices)})"
        if args.three_d:
            ax.scatter(coords[indices, 0], coords[indices, 1], coords[indices, 2],
                       s=35, c=color, alpha=0.9, label=label)
        else:
            ax.scatter(coords[indices, 0], coords[indices, 1],
                       s=35, c=color, alpha=0.9, label=label)

    if args.annotate:
        for idx in ordered_ids:
            p = coords[id_to_index[idx]]
            if args.three_d:
                ax.text(p[0], p[1], p[2], str(idx), fontsize=6, color="black")
            else:
                ax.text(p[0], p[1], str(idx), fontsize=6, color="black")

    first_idx = id_to_index[first_node_id]
    first_pt = coords[first_idx]
    if args.three_d:
        ax.scatter([first_pt[0]], [first_pt[1]], [first_pt[2]], s=80, c="#ff7f0e",
                   edgecolors="black", marker="*", label="First node")
    else:
        ax.scatter([first_pt[0]], [first_pt[1]], s=80, c="#ff7f0e",
                   edgecolors="black", marker="*", label="First node")

    def draw_edges(edge_list: EdgeList, color: str, label: str, alpha: float, linewidth: float):
        segments, _ = _edge_segments(edge_list, coords, id_to_index)
        if not len(segments):
            return
        if args.three_d:
            for seg in segments:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, linewidth=linewidth)
        else:
            for seg in segments:
                ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=alpha, linewidth=linewidth)
        ax.plot([], [], color=color, label=label, linewidth=linewidth, alpha=alpha)

    draw_edges(odom_edges, color="#1f77b4", label=f"Odometry ({len(odom_edges)})", alpha=0.5, linewidth=1.0)
    draw_edges(loop_edges, color="#d62728", label=f"Loop ({len(loop_edges)})", alpha=0.9, linewidth=1.5)

    def draw_prior_lines(nodes: Set[int]) -> None:
        if not nodes:
            return
        segments = []
        for idx in nodes:
            if idx not in id_to_index:
                continue
            node_pt = coords[id_to_index[idx]]
            origin = np.zeros_like(node_pt)
            segments.append(np.vstack([origin, node_pt]))
        if not segments:
            return
        segments_np = np.stack(segments)
        if args.three_d:
            for seg in segments_np:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#9467bd", linestyle="--", linewidth=1.0, alpha=0.8)
        else:
            for seg in segments_np:
                ax.plot(seg[:, 0], seg[:, 1], color="#9467bd", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.plot([], [], color="#9467bd", linestyle="--", linewidth=1.0, label=f"Pose priors ({len(nodes)})")

    draw_prior_lines(prior_nodes)

    ax.set_title(f"Pose graph: {len(vertices)} nodes, {len(odom_edges)} odom, {len(loop_edges)} loops")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = args.g2o.parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a pose graph stored in g2o format.")
    parser.add_argument("--g2o", required=True, type=Path, help="Path to pose_graph.g2o file")
    parser.add_argument("--odom-threshold", type=int, default=1,
                        help="Edges with |id_i - id_j| <= threshold are treated as odometry (default: 1)")
    parser.add_argument("--output", type=str, default="pose_graph.png",
                        help="File name or path for the plot. Relative paths are saved next to the g2o file.")
    parser.add_argument("--show", action="store_true", default=False, help="Display interactive window with the plot")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Save only, no interactive window")
    parser.add_argument("--three-d", action="store_true", help="Use a 3D view instead of XY top-down")
    parser.add_argument("--annotate", action="store_true", help="Annotate node IDs in the plot")

    args = parser.parse_args()

    if not args.g2o.exists():
        raise FileNotFoundError(f"g2o file not found: {args.g2o}")

    vertices, edges, prior_nodes, gnss_nodes = parse_g2o(args.g2o)
    odom_edges, loop_edges = classify_edges(edges, args.odom_threshold)

    print(f"Loaded {len(vertices)} vertices and {len(edges)} edges from {args.g2o}")
    print(f"Classified {len(odom_edges)} odometry edges and {len(loop_edges)} loop edges")
    if prior_nodes:
        print(f"Detected {len(prior_nodes)} pose prior constraints")
    if gnss_nodes:
        counts = {}
        for constraint in gnss_nodes.values():
            counts[constraint] = counts.get(constraint, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"Detected GNSS-constrained nodes -> {summary}")

    plot_pose_graph(vertices, odom_edges, loop_edges, prior_nodes, gnss_nodes, args)


if __name__ == "__main__":
    main()


# python3 visualize_pose_graph.py --g2o '/home/auto/data/ms_mapping_result/rosbag2_2025_09_29-14_09_14/pose_graph.g2o'
