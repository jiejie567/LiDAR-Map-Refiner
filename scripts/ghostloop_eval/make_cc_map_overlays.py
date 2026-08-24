#!/usr/bin/env python3
"""Crop two exported maps into colored local overlays for CloudCompare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gui"))
from merge_pcds import read_pcd


def xyz(path: Path) -> np.ndarray:
    data = read_pcd(path).data
    return np.column_stack([data["x"], data["y"], data["z"]])


def write_overlay(path: Path, old: np.ndarray, new: np.ndarray) -> None:
    points = np.vstack([old, new])
    colors = np.vstack([
        np.tile([255, 40, 40], (len(old), 1)),
        np.tile([0, 255, 255], (len(new), 1)),
    ])
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{color[0]} {color[1]} {color[2]}\n")


def crop(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return points[np.all(np.abs(points - center) <= radius, axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_12cm", type=Path)
    parser.add_argument("map_20cm", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old, new = xyz(args.map_12cm), xyz(args.map_20cm)
    regions = json.loads(args.report.read_text(encoding="utf-8")).get(
        "ghost_regions", [])
    centers = [
        (f"ghost_{i:02d}_{round(r['separation_m']*100):02d}cm",
         np.asarray(r["center_xyz"], dtype=float), 2.5)
        for i, r in enumerate(regions, 1)
    ]
    centers.append(("MAX_MOTION_kf332_87cm",
                    np.asarray([-1.73, 3.42, 0.09]), 4.0))
    for name, center, radius in centers:
        write_overlay(
            args.output_dir / f"{name}_RED12_CYAN20.ply",
            crop(old, center, radius),
            crop(new, center, radius),
        )


if __name__ == "__main__":
    main()
