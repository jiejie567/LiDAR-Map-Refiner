#!/usr/bin/env python3
"""Create CloudCompare-visible colored wire boxes from a BALM ghost report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def box_points(center: np.ndarray, normal: np.ndarray, separation: float) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent /= np.linalg.norm(tangent)
    vertical = np.cross(normal, tangent)
    # Thin across the diagnosed layer normal, broad along the surface.
    half = np.array([max(0.6, separation * 2.5), 2.0, 1.5])
    basis = np.stack([normal, tangent, vertical], axis=1)
    corners = np.array(
        [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
        dtype=float,
    )
    corners = center + (corners * half) @ basis.T
    edges = []
    for i in range(8):
        for axis in range(3):
            j = i ^ (1 << axis)
            if i < j:
                t = np.linspace(0.0, 1.0, 80)[:, None]
                edges.append(corners[i] * (1.0 - t) + corners[j] * t)
    return np.vstack(edges)


def write_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for point in points:
            stream.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions = json.loads(args.report.read_text(encoding="utf-8")).get(
        "ghost_regions", [])
    combined = []
    for index, region in enumerate(regions, 1):
        points = box_points(
            np.asarray(region["center_xyz"], dtype=float),
            np.asarray(region["normal"], dtype=float),
            float(region["separation_m"]),
        )
        combined.append(points)
        severity_cm = round(float(region["separation_m"]) * 100)
        pair = region.get("suggested_pair", ["x", "x"])
        name = (
            f"ghost_{index:02d}_{severity_cm:02d}cm_"
            f"kf{pair[0]}-{pair[1]}.ply"
        )
        color = (255, 0, 255) if index == 7 else (255, 32, 32)
        write_ply(args.output_dir / name, points, color)
    if combined:
        write_ply(args.output_dir / "ALL_GHOST_BOXES.ply",
                  np.vstack(combined), (255, 32, 32))


if __name__ == "__main__":
    main()
