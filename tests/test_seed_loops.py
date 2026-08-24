from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Leaf-module imports (matching the balm tests) so the test runs without open3d.
PKG_DIR = Path(__file__).resolve().parents[1] / "gui" / "manual_loop_closure"
sys.path.insert(0, str(PKG_DIR))

from scan_context_io import ScanContextConfig  # noqa: E402
from seed_loops import (  # noqa: E402
    _descriptor_distance,
    build_descriptor_and_prepared_stack,
    build_descriptor_stack,
    find_seed_candidates,
    ghost_badness,
    pair_yaw_alignment,
)


def _rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _local_structure(center: np.ndarray, seed: int) -> np.ndarray:
    """Distinctive structure around one place: random pillars + one wall."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(4):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(3.0, 10.0)
        cx, cy = center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)
        parts.append(
            np.column_stack(
                [
                    cx + 0.3 * np.cos(rng.uniform(0, 2 * np.pi, 250)),
                    cy + 0.3 * np.sin(rng.uniform(0, 2 * np.pi, 250)),
                    rng.uniform(0.0, 3.0, 250),
                ]
            )
        )
    wall_angle = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(wall_angle), np.sin(wall_angle)])
    offsets = rng.uniform(-8.0, 8.0, 800)
    wall_center = center[:2] + rng.uniform(4.0, 8.0) * np.array(
        [np.cos(wall_angle + np.pi / 2), np.sin(wall_angle + np.pi / 2)]
    )
    parts.append(
        np.column_stack(
            [
                wall_center[0] + offsets * direction[0],
                wall_center[1] + offsets * direction[1],
                rng.uniform(0.0, 3.0, 800),
            ]
        )
    )
    return np.vstack(parts)


class SeedLoopTest(unittest.TestCase):
    def test_accidental_one_sector_match_is_not_perfect(self) -> None:
        descriptors = np.zeros((2, 20, 60), dtype=np.float64)
        masks = np.zeros_like(descriptors, dtype=bool)
        # Each scan has broad support, but only one sector overlaps. The old
        # Python scorer dropped every unsupported sector and returned d=0.
        masks[0, :3, :10] = True
        masks[1, :3, 0] = True
        masks[1, :3, 20:30] = True
        descriptors[masks] = 2.0

        distance = _descriptor_distance(
            descriptors[0],
            masks[0],
            descriptors[1],
            masks[1],
            channel_weights=None,
            num_rings=20,
            min_joint_rings=2,
            retrieval_height_offset=0.1,
            sector_support_exponent=0.5,
        )

        self.assertGreater(distance, 0.5)

    def test_full_sector_support_keeps_identical_match(self) -> None:
        descriptors = np.zeros((2, 20, 60), dtype=np.float64)
        masks = np.zeros_like(descriptors, dtype=bool)
        masks[:, :3, :12] = True
        descriptors[masks] = 2.0

        distance, _ = pair_yaw_alignment(
            descriptors, masks, 0, 1, num_rings=20)

        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_combined_build_reads_each_cloud_once(self) -> None:
        clouds = [np.full((4, 3), float(i)) for i in range(5)]
        reads = [0] * len(clouds)
        progress = []

        def load(index: int) -> np.ndarray:
            reads[index] += 1
            return clouds[index]

        config = ScanContextConfig(gravity_canonicalization_enable=False)
        descriptors, masks, prepared = build_descriptor_and_prepared_stack(
            load,
            lambda points: points[:2] + 10.0,
            len(clouds),
            config,
            progress_fn=lambda current, total: progress.append((current, total)),
        )

        self.assertEqual(reads, [1] * len(clouds))
        self.assertEqual(
            progress,
            [(index, len(clouds)) for index in range(1, len(clouds) + 1)],
        )
        self.assertEqual(descriptors.shape[0], len(clouds))
        self.assertEqual(masks.shape[0], len(clouds))
        for index, cloud in enumerate(prepared):
            np.testing.assert_array_equal(cloud, clouds[index][:2] + 10.0)

    def _make_frames(self, revisit_yaw_deg: float):
        # Places spaced far apart, each with unique structure; frame 10 revisits
        # frame 2's place with a yaw change. max_radius=15 keeps views local.
        positions = [np.array([40.0 * k, 0.0, 0.0]) for k in range(12)]
        positions[10] = positions[2].copy()
        world = np.vstack(
            [
                _local_structure(positions[k], seed=k)
                for k in range(12)
                if k != 10
            ]
        )
        yaws = [0.0] * 12
        yaws[10] = revisit_yaw_deg
        clouds = []
        for pos, yaw in zip(positions, yaws):
            rotation = _rot_z(yaw)
            clouds.append((world - pos) @ rotation)  # world -> body frame
        return clouds

    def test_finds_revisit_pair_and_yaw(self) -> None:
        revisit_yaw = 40.0
        clouds = self._make_frames(revisit_yaw)
        config = ScanContextConfig(gravity_canonicalization_enable=False, max_radius=15.0)
        descriptors, masks = build_descriptor_stack(
            lambda i: clouds[i], len(clouds), config
        )
        seeds = find_seed_candidates(
            descriptors,
            masks,
            min_index_gap=4,
            max_seeds=3,
            max_distance=0.45,
        )
        self.assertTrue(seeds, "no seed candidate found")
        top = seeds[0]
        self.assertEqual((top.source_id, top.target_id), (10, 2))
        # Convention check: rotating the source cloud by +yaw_deg about z must
        # reproduce the target's descriptor (near-zero residual shift).
        rotated = clouds[10] @ _rot_z(top.yaw_deg)
        from scan_context_io import make_descriptor_with_mask

        desc_rot, mask_rot = make_descriptor_with_mask(rotated, config)
        desc_tgt, mask_tgt = make_descriptor_with_mask(clouds[2], config)
        both = mask_rot & mask_tgt
        num = float(np.sum(desc_rot[both] * desc_tgt[both]))
        den = float(
            np.linalg.norm(desc_rot[both]) * np.linalg.norm(desc_tgt[both])
        )
        self.assertGreater(num / max(den, 1e-9), 0.9,
                           f"yaw convention wrong: yaw={top.yaw_deg:.1f}")
        self.assertLessEqual(
            abs(abs(top.yaw_deg) - revisit_yaw), 9.0,
            f"yaw estimate {top.yaw_deg:.1f} deg off from {revisit_yaw}",
        )

    def test_no_false_seed_without_revisit(self) -> None:
        clouds = self._make_frames(0.0)
        clouds[10] = clouds[10] + np.array([5000.0, 5000.0, 0.0])  # see nothing
        config = ScanContextConfig(gravity_canonicalization_enable=False, max_radius=15.0)
        descriptors, masks = build_descriptor_stack(
            lambda i: clouds[i], len(clouds), config
        )
        seeds = find_seed_candidates(
            descriptors, masks, min_index_gap=4, max_seeds=3, max_distance=0.15
        )
        pairs = {(s.source_id, s.target_id) for s in seeds}
        self.assertNotIn((10, 2), pairs)

    def test_ghost_badness(self) -> None:
        regions = [
            {"point_count": 100, "separation_m": 0.2},
            {"point_count": 50, "separation_m": 0.1},
        ]
        self.assertAlmostEqual(ghost_badness(regions), 25.0)

    def test_ghost_badness_uses_component_typical_separation(self) -> None:
        regions = [{
            "point_count": 100,
            "separation_m": 0.30,
            "separation_typical_m": 0.12,
        }]
        self.assertAlmostEqual(ghost_badness(regions), 12.0)


if __name__ == "__main__":
    unittest.main()
