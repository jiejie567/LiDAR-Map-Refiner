from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TOOL_DIR = Path(__file__).resolve().parents[1]
GUI_DIR = TOOL_DIR / "gui"
OPTIMIZER_DIR = GUI_DIR / "manual_loop_closure" / "python_optimizer"
for path in (TOOL_DIR, GUI_DIR, OPTIMIZER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Imported as a leaf module (matching balm_cli.py) so the test runs without open3d.
from balm import (  # noqa: E402
    BalmParams,
    _joint_plane_system,
    _merge_connected_ghost_voxels,
    _rms_has_plateaued,
    _solve_damped_system,
    detect_ghost_regions,
    extract_planar_voxels,
    run_balm_refinement,
)
from balm_cli import _build_parser, _termination_summary  # noqa: E402


def _sample_room_points(rng: np.random.Generator, count_per_plane: int) -> np.ndarray:
    """World points on three orthogonal planes of a 8x8x3 m room with 1 cm noise."""
    floor = np.column_stack(
        [
            rng.uniform(0.0, 8.0, count_per_plane),
            rng.uniform(0.0, 8.0, count_per_plane),
            rng.normal(0.0, 0.01, count_per_plane),
        ]
    )
    wall_x = np.column_stack(
        [
            rng.normal(0.0, 0.01, count_per_plane),
            rng.uniform(0.0, 8.0, count_per_plane),
            rng.uniform(0.0, 3.0, count_per_plane),
        ]
    )
    wall_y = np.column_stack(
        [
            rng.uniform(0.0, 8.0, count_per_plane),
            rng.normal(0.0, 0.01, count_per_plane),
            rng.uniform(0.0, 3.0, count_per_plane),
        ]
    )
    return np.vstack([floor, wall_x, wall_y])


def _make_pose(yaw_deg: float, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


def _pose_errors(poses_a: np.ndarray, poses_b: np.ndarray) -> tuple[float, float]:
    trans_err = float(
        np.max(np.linalg.norm(poses_a[:, :3, 3] - poses_b[:, :3, 3], axis=1))
    )
    rot_err = 0.0
    for index in range(poses_a.shape[0]):
        delta = poses_a[index, :3, :3].T @ poses_b[index, :3, :3]
        rot_err = max(rot_err, float(np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())))
    return trans_err, rot_err


class BalmRefinementTest(unittest.TestCase):
    def test_rms_increase_is_not_misreported_as_plateau(self) -> None:
        self.assertFalse(_rms_has_plateaued([0.0661, 0.0673, 0.0677], 2, 5e-5))
        self.assertTrue(_rms_has_plateaued([0.031859, 0.031830, 0.031819], 2, 5e-5))
        self.assertFalse(_rms_has_plateaued([0.0319, 0.0317, 0.0316], 2, 5e-5))

    def test_large_system_iterative_backend_matches_direct_solution(self) -> None:
        diagonal = np.full(40, 4.0)
        off_diagonal = np.full(39, -1.0)
        from scipy import sparse

        matrix = sparse.diags(
            [off_diagonal, diagonal, off_diagonal], [-1, 0, 1], format="csr"
        )
        rhs = np.linspace(-2.0, 3.0, 40)
        direct, direct_name, _, _ = _solve_damped_system(
            matrix, rhs, direct_max_variables=100
        )
        iterative, iterative_name, iterations, relative_residual = (
            _solve_damped_system(matrix, rhs, direct_max_variables=0)
        )

        self.assertEqual(direct_name, "sparse_direct")
        self.assertEqual(iterative_name, "cg_jacobi")
        self.assertGreater(iterations, 0)
        self.assertLess(relative_residual, 1e-6)
        np.testing.assert_allclose(iterative, direct, rtol=1e-5, atol=1e-8)

    def test_production_defaults_are_unregularized_and_convergence_driven(self) -> None:
        params = BalmParams()
        self.assertEqual(params.max_iterations, 20)
        self.assertEqual(params.pose_prior_translation_weight, 0.0)
        self.assertEqual(params.pose_prior_rotation_weight, 0.0)
        self.assertTrue(params.double_sided_enable)
        cli = _build_parser().parse_args(
            ["--tum", "input.tum", "--keyframe-dir", "pcd", "--output-dir", "out"]
        )
        self.assertEqual(cli.max_iterations, 20)
        self.assertEqual(cli.pose_prior_translation_weight, 0.0)
        self.assertEqual(cli.pose_prior_rotation_weight, 0.0)
        self.assertTrue(cli.double_sided)
        self.assertFalse(cli.diagnose_map_inconsistency)

    def test_configured_update_cap_is_not_reported_as_solver_failure(self) -> None:
        summary = _termination_summary(
            single_stage=False,
            stage_plan=[(0.12, 1), (0.05, 1)],
            stage_reports=[
                {"converged": False, "stop_reason": "iteration_limit", "iterations": [{}]},
                {"converged": False, "stop_reason": "iteration_limit", "iterations": [{}]},
            ],
        )
        self.assertEqual(summary["run_status"], "completed")
        self.assertEqual(summary["policy"], "coarse_to_fine_convergence")
        self.assertEqual(summary["reason"], "configured_update_cap_reached")
        self.assertEqual(summary["requested_updates"], 2)
        self.assertEqual(summary["completed_updates"], 2)
        self.assertFalse(summary["final_stage_numerically_converged"])
        self.assertFalse(summary["all_stages_numerically_converged"])

    def test_objective_plateau_is_not_reported_as_pose_convergence(self) -> None:
        summary = _termination_summary(
            single_stage=True,
            stage_plan=[(0.05, 4)],
            stage_reports=[
                {
                    "converged": False,
                    "stop_reason": "objective_plateau",
                    "iterations": [{}, {}],
                },
            ],
        )
        self.assertEqual(summary["policy"], "single_stage_convergence")
        self.assertEqual(summary["reason"], "final_stage_objective_plateau")
        self.assertEqual(summary["completed_updates"], 2)
        self.assertFalse(summary["final_stage_numerically_converged"])
        self.assertTrue(summary["final_stage_objective_plateau"])
        self.assertFalse(summary["all_stages_numerically_converged"])

    def test_pose_update_convergence_is_reported_explicitly(self) -> None:
        summary = _termination_summary(
            single_stage=True,
            stage_plan=[(0.05, 4)],
            stage_reports=[
                {
                    "converged": True,
                    "stop_reason": "pose_converged",
                    "iterations": [{}],
                },
            ],
        )
        self.assertEqual(summary["reason"], "final_stage_pose_converged")
        self.assertTrue(summary["final_stage_pose_converged"])
        self.assertFalse(summary["final_stage_objective_plateau"])
        self.assertTrue(summary["all_stages_pose_converged"])

    def test_final_stage_plateau_is_distinct_from_coarse_stage_cap(self) -> None:
        summary = _termination_summary(
            single_stage=False,
            stage_plan=[(0.12, 10), (0.05, 10)],
            stage_reports=[
                {"converged": False, "stop_reason": "iteration_limit", "iterations": [{}] * 10},
                {
                    "converged": False,
                    "stop_reason": "objective_plateau",
                    "iterations": [{}] * 3,
                },
            ],
        )
        self.assertEqual(summary["reason"], "final_stage_objective_plateau")
        self.assertFalse(summary["final_stage_numerically_converged"])
        self.assertTrue(summary["final_stage_objective_plateau"])
        self.assertFalse(summary["all_stages_numerically_converged"])

    def _build_scene(self):
        rng = np.random.default_rng(7)
        world_points = _sample_room_points(rng, 900)
        gt_poses = np.stack(
            [
                _make_pose(0.0, np.array([2.0, 2.0, 1.0])),
                _make_pose(25.0, np.array([4.0, 2.5, 1.2])),
                _make_pose(-30.0, np.array([5.5, 4.5, 0.8])),
                _make_pose(60.0, np.array([3.0, 5.5, 1.5])),
            ]
        )
        clouds = []
        for pose in gt_poses:
            local = (world_points - pose[:3, 3]) @ pose[:3, :3]
            clouds.append(local)
        return rng, gt_poses, clouds

    def _build_double_sided_scene(self):
        """A 12 cm partition observed independently from both physical sides."""
        rng = np.random.default_rng(21)
        count = 700

        def sheet(x_position):
            return np.column_stack([
                rng.normal(x_position, 0.005, count),
                rng.uniform(0.0, 6.0, count),
                rng.uniform(0.6, 3.0, count),
            ])

        def floor(x_low, x_high):
            return np.column_stack([
                rng.uniform(x_low, x_high, count),
                rng.uniform(0.0, 6.0, count),
                rng.normal(0.5, 0.005, count),
            ])

        face_positive = sheet(0.56)
        face_negative = sheet(0.44)
        poses = np.stack([
            _make_pose(180.0, np.array([2.5, 2.0, 1.5])),
            _make_pose(150.0, np.array([2.5, 4.0, 1.5])),
            _make_pose(0.0, np.array([-2.5, 2.0, 1.5])),
            _make_pose(-30.0, np.array([-2.5, 4.0, 1.5])),
        ])
        visible = [
            [face_positive, floor(0.8, 4.2)],
            [face_positive, floor(0.8, 4.2)],
            [face_negative, floor(-4.2, 0.2)],
            [face_negative, floor(-4.2, 0.2)],
        ]
        clouds, labels = [], []
        for pose, planes in zip(poses, visible):
            world = np.vstack(planes)
            clouds.append((world - pose[:3, 3]) @ pose[:3, :3])
            label = np.zeros(len(world), dtype=np.int8)
            label[:count] = 1 if planes[0] is face_positive else 2
            labels.append(label)
        return poses, clouds, labels

    def test_extract_planar_voxels_finds_planes(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        params = BalmParams(root_voxel_size=1.0, downsample_leaf=0.0)
        world = np.vstack(
            [cloud @ pose[:3, :3].T + pose[:3, 3] for cloud, pose in zip(clouds, gt_poses)]
        )
        pose_ids = np.repeat(np.arange(len(clouds)), [c.shape[0] for c in clouds])
        planes = extract_planar_voxels(world, pose_ids, params)
        self.assertGreater(len(planes), 20)

    def test_oriented_plane_extraction_never_mixes_partition_faces(self) -> None:
        poses, clouds, labels = self._build_double_sided_scene()
        world = np.vstack([
            cloud @ pose[:3, :3].T + pose[:3, 3]
            for cloud, pose in zip(clouds, poses)
        ])
        pose_ids = np.repeat(
            np.arange(len(clouds)), [len(cloud) for cloud in clouds]
        )
        sensors = poses[pose_ids, :3, 3]
        label = np.concatenate(labels)
        diagnostics = []
        planes = extract_planar_voxels(
            world, pose_ids,
            BalmParams(
                root_voxel_size=1.0, downsample_leaf=0.0,
                plane_thickness=0.12, double_sided_enable=True,
            ),
            sensors,
            diagnostics,
        )
        self.assertTrue(planes)
        self.assertTrue(diagnostics)
        self.assertTrue(any(0.09 < item["separation_m"] < 0.15 for item in diagnostics))
        mixed = sum(
            (label[rows] == 1).any() and (label[rows] == 2).any()
            for rows in planes
        )
        self.assertEqual(mixed, 0)

    def test_oriented_balm_preserves_partition_thickness(self) -> None:
        poses, clouds, labels = self._build_double_sided_scene()
        rng = np.random.default_rng(5)
        perturbed = poses.copy()
        for index in range(1, len(perturbed)):
            perturbed[index, :3, :3] = (
                perturbed[index, :3, :3]
                @ Rotation.from_rotvec(rng.normal(0.0, 0.008, 3)).as_matrix()
            )
            perturbed[index, :3, 3] += rng.normal(0.0, 0.03, 3)

        def thickness(result_poses):
            faces = {1: [], 2: []}
            for pose, cloud, label in zip(result_poses, clouds, labels):
                world = cloud @ pose[:3, :3].T + pose[:3, 3]
                for face in (1, 2):
                    if (label == face).any():
                        faces[face].append(world[label == face, 0])
            return float(
                np.concatenate(faces[1]).mean()
                - np.concatenate(faces[2]).mean()
            )

        common = dict(
            root_voxel_size=1.0, downsample_leaf=0.0,
            plane_thickness=0.12, max_iterations=10,
            reassociate_every=3,
        )
        guarded = run_balm_refinement(
            clouds, perturbed,
            BalmParams(**common, double_sided_enable=True),
        )
        unguarded = run_balm_refinement(
            clouds, perturbed,
            BalmParams(**common, double_sided_enable=False),
        )
        guarded_thickness = thickness(guarded.poses_world_sensor)
        unguarded_thickness = thickness(unguarded.poses_world_sensor)
        self.assertGreater(guarded_thickness, 0.09)
        self.assertLess(unguarded_thickness, guarded_thickness - 0.02)
        self.assertTrue(any(
            item.double_sided_voxel_count > 0 for item in guarded.iterations
        ))
        self.assertTrue(all(
            item.double_sided_voxel_count == 0 for item in unguarded.iterations
        ))

    def test_production_two_stage_schedule_keeps_opposite_faces_separate(self) -> None:
        poses, clouds, labels = self._build_double_sided_scene()
        rng = np.random.default_rng(5)
        perturbed = poses.copy()
        for index in range(1, len(perturbed)):
            perturbed[index, :3, :3] = (
                perturbed[index, :3, :3]
                @ Rotation.from_rotvec(rng.normal(0.0, 0.008, 3)).as_matrix()
            )
            perturbed[index, :3, 3] += rng.normal(0.0, 0.03, 3)

        def solve(enabled: bool) -> np.ndarray:
            result_poses = perturbed.copy()
            for thickness in (0.12, 0.05):
                result_poses = run_balm_refinement(
                    clouds,
                    result_poses,
                    BalmParams(
                        root_voxel_size=1.0,
                        downsample_leaf=0.0,
                        plane_thickness=thickness,
                        max_iterations=1,
                        double_sided_enable=enabled,
                    ),
                ).poses_world_sensor
            return result_poses

        def thickness(result_poses: np.ndarray) -> float:
            face_points = {1: [], 2: []}
            for pose, cloud, label in zip(result_poses, clouds, labels):
                world = cloud @ pose[:3, :3].T + pose[:3, 3]
                for face in (1, 2):
                    if np.any(label == face):
                        face_points[face].append(world[label == face, 0])
            return float(
                np.concatenate(face_points[1]).mean()
                - np.concatenate(face_points[2]).mean()
            )

        guarded = thickness(solve(True))
        unguarded = thickness(solve(False))
        self.assertGreater(guarded, 0.09)
        self.assertLess(abs(unguarded), 0.01)

    def test_pose_graph_trust_prevents_disconnected_side_drift(self) -> None:
        poses, clouds, labels = self._build_double_sided_scene()
        rng = np.random.default_rng(43)
        perturbed = poses.copy()
        for index in range(1, len(perturbed)):
            perturbed[index, :3, 3] += rng.normal(0.0, 0.02, 3)

        params = BalmParams(
            root_voxel_size=1.0, downsample_leaf=0.0,
            plane_thickness=0.12, max_iterations=12,
            double_sided_enable=True,
            pose_prior_translation_weight=300.0,
            pose_prior_rotation_weight=1200.0,
        )
        result = run_balm_refinement(clouds, perturbed, params)
        total_motion = np.linalg.norm(
            result.poses_world_sensor[:, :3, 3] - perturbed[:, :3, 3], axis=1
        )
        self.assertLess(float(total_motion.max()), 0.08)

    def test_refinement_recovers_perturbed_poses(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        perturbed = gt_poses.copy()
        for index in range(1, perturbed.shape[0]):
            noise_rot = Rotation.from_rotvec(rng.normal(0.0, 0.01, 3)).as_matrix()
            perturbed[index, :3, :3] = perturbed[index, :3, :3] @ noise_rot
            perturbed[index, :3, 3] = perturbed[index, :3, 3] + rng.normal(0.0, 0.05, 3)

        initial_trans, initial_rot = _pose_errors(gt_poses, perturbed)
        params = BalmParams(
            root_voxel_size=1.0,
            downsample_leaf=0.05,
            max_iterations=15,
            reassociate_every=3,
        )
        result = run_balm_refinement(clouds, perturbed, params)
        final_trans, final_rot = _pose_errors(gt_poses, result.poses_world_sensor)

        self.assertTrue(result.iterations, "BALM ran no iterations")
        self.assertLess(final_trans, initial_trans * 0.35)
        self.assertLess(final_rot, initial_rot * 0.5)
        self.assertLess(
            result.iterations[-1].rms_plane_distance,
            result.iterations[0].rms_plane_distance,
        )
        # Gauge pose must stay untouched.
        np.testing.assert_allclose(result.poses_world_sensor[0], perturbed[0], atol=1e-12)

    def test_plane_elimination_couples_observing_poses(self) -> None:
        """One shared plane must create off-diagonal pose Hessian blocks."""
        rng = np.random.default_rng(101)
        points_per_pose = 30
        pose_ids = np.repeat(np.arange(3), points_per_pose)
        local = rng.normal(size=(pose_ids.size, 3))
        local[:, 2] *= 0.01
        rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        translations = np.zeros((3, 3))
        translations[:, 2] = [0.0, 0.03, -0.02]
        world = local + translations[pose_ids]
        plane_ids = np.zeros(pose_ids.size, dtype=np.int64)
        centroid = world.mean(axis=0, keepdims=True)
        covariance = np.cov(world.T, bias=True)
        _, eigenvectors = np.linalg.eigh(covariance)
        eigenvectors = eigenvectors[None, :, :]
        normals = eigenvectors[:, :, 0]
        residuals = np.einsum(
            "mi,mi->m", world - centroid, normals[plane_ids]
        )
        hessian, _ = _joint_plane_system(
            local,
            pose_ids,
            plane_ids,
            world,
            rotations,
            centroid,
            eigenvectors,
            normals,
            residuals,
            np.ones_like(residuals),
        )
        cross_block = hessian[:6, 6:12].toarray()
        self.assertGreater(float(np.linalg.norm(cross_block)), 1e-6)

    def test_lm_acceptance_keeps_fixed_association_objective_monotone(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        perturbed = gt_poses.copy()
        for index in range(1, perturbed.shape[0]):
            perturbed[index, :3, 3] += rng.normal(0.0, 0.05, 3)
        result = run_balm_refinement(
            clouds,
            perturbed,
            BalmParams(
                root_voxel_size=1.0,
                downsample_leaf=0.05,
                max_iterations=8,
                reassociate_every=0,
            ),
        )
        objectives = [item.objective for item in result.iterations]
        self.assertTrue(all(item.accepted for item in result.iterations))
        self.assertTrue(
            all(after <= before + 1e-12 for before, after in zip(objectives, objectives[1:]))
        )

    def test_reassociation_resets_decayed_lm_trust_region(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        perturbed = gt_poses.copy()
        for index in range(1, perturbed.shape[0]):
            perturbed[index, :3, 3] += rng.normal(0.0, 0.04, 3)
        params = BalmParams(
            root_voxel_size=1.0,
            downsample_leaf=0.05,
            max_iterations=5,
            reassociate_every=1,
            reassociate_min_motion=0.0,
            lm_damping=1e-2,
            plateau_window=0,
        )
        result = run_balm_refinement(clouds, perturbed, params)
        reassociated = [item for item in result.iterations if item.reassociated]
        self.assertGreaterEqual(len(reassociated), 2)
        self.assertTrue(all(
            item.lm_damping >= params.lm_damping for item in reassociated
        ))

    def test_robust_kernel_tolerates_clutter(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        # Simulate clutter/dynamic objects: dense random blobs in two keyframes.
        for index in (1, 3):
            blob = rng.uniform(-1.0, 1.0, (400, 3)) * np.array([0.5, 0.5, 0.4]) + np.array(
                [1.0, 1.5, 0.6]
            )
            clouds[index] = np.vstack([clouds[index], blob])
        perturbed = gt_poses.copy()
        for index in range(1, perturbed.shape[0]):
            noise_rot = Rotation.from_rotvec(rng.normal(0.0, 0.01, 3)).as_matrix()
            perturbed[index, :3, :3] = perturbed[index, :3, :3] @ noise_rot
            perturbed[index, :3, 3] = perturbed[index, :3, 3] + rng.normal(0.0, 0.05, 3)
        initial_trans, _ = _pose_errors(gt_poses, perturbed)
        params = BalmParams(
            root_voxel_size=1.0,
            downsample_leaf=0.05,
            max_iterations=15,
            robust_kernel="huber",
        )
        result = run_balm_refinement(clouds, perturbed, params)
        final_trans, _ = _pose_errors(gt_poses, result.poses_world_sensor)
        self.assertLess(final_trans, initial_trans * 0.4)

    def test_plateau_stops_early(self) -> None:
        rng, gt_poses, clouds = self._build_scene()
        perturbed = gt_poses.copy()
        perturbed[1, :3, 3] = perturbed[1, :3, 3] + np.array([0.03, -0.02, 0.01])
        params = BalmParams(
            root_voxel_size=1.0,
            downsample_leaf=0.05,
            max_iterations=60,
        )
        result = run_balm_refinement(clouds, perturbed, params)
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "objective_plateau")
        self.assertLess(len(result.iterations), 60)
        last = result.iterations[-1]
        self.assertGreaterEqual(last.max_translation_update_m, 0.0)
        self.assertGreaterEqual(last.max_rotation_update_deg, 0.0)
        report = result.report_dict()["iterations"][-1]
        self.assertIn("max_translation_update_m", report)
        self.assertIn("max_rotation_update_deg", report)

    def test_ghost_detection_reports_pass_pair(self) -> None:
        rng = np.random.default_rng(9)
        n = 700
        wall = np.column_stack(
            [
                rng.normal(0.5, 0.005, n),
                rng.uniform(0.0, 6.0, n),
                rng.uniform(0.6, 3.0, n),
            ]
        )
        floor = np.column_stack(
            [
                rng.uniform(0.8, 4.2, n),
                rng.uniform(0.0, 6.0, n),
                rng.normal(0.5, 0.005, n),
            ]
        )
        true_poses = np.stack(
            [
                _make_pose(180.0, np.array([3.0, 1.5, 1.5])),
                _make_pose(170.0, np.array([3.0, 3.0, 1.5])),
                _make_pose(190.0, np.array([3.0, 4.5, 1.5])),
                _make_pose(180.0, np.array([3.0, 6.0, 1.5])),
            ]
        )
        clouds = [
            (np.vstack([wall, floor]) - pose[:3, 3]) @ pose[:3, :3]
            for pose in true_poses
        ]
        # Registration error: the second pass (kf 2-3) is shifted 15 cm along x,
        # creating a same-side double layer of the wall.
        wrong_poses = true_poses.copy()
        wrong_poses[2:, 0, 3] += 0.15
        params = BalmParams(root_voxel_size=1.0, downsample_leaf=0.0)
        # This compact unit scene has only two poses per traversal and one
        # populated root voxel in places; production defaults require broader
        # multi-frame / spatial support.
        regions = detect_ghost_regions(
            clouds,
            wrong_poses,
            params,
            min_layer_keyframes=2,
            min_region_voxels=1,
        )
        self.assertTrue(regions, "ghost layer not detected")
        top = regions[0]
        self.assertGreater(top["separation_m"], 0.10)
        self.assertLess(top["separation_m"], 0.20)
        self.assertEqual(top["candidate_type"], "duplicated_surface_region")
        self.assertGreater(top["overlap_area_m2"], 0.0)
        self.assertGreater(top["surface_displacement_m3"], 0.0)
        self.assertEqual(
            top["loop_observability"], "single_plane_underconstrained"
        )
        self.assertFalse(top["loop_constraint_ready"])
        pair = sorted(top["suggested_pair"])
        self.assertLessEqual(pair[0], 1)
        self.assertGreaterEqual(pair[1], 2)

    def test_ghost_detection_ignores_opposite_wall_faces(self) -> None:
        rng = np.random.default_rng(11)
        n = 900
        face_left = np.column_stack(
            [
                rng.normal(0.44, 0.004, n),
                rng.uniform(0.0, 4.0, n),
                rng.uniform(0.5, 2.5, n),
            ]
        )
        face_right = face_left.copy()
        face_right[:, 0] += 0.14
        poses = np.stack(
            [
                _make_pose(0.0, np.array([-2.0, 1.0, 1.2])),
                _make_pose(0.0, np.array([-2.0, 3.0, 1.2])),
                _make_pose(180.0, np.array([3.0, 1.0, 1.2])),
                _make_pose(180.0, np.array([3.0, 3.0, 1.2])),
            ]
        )
        clouds = []
        for index, pose in enumerate(poses):
            observed_face = face_left if index < 2 else face_right
            clouds.append((observed_face - pose[:3, 3]) @ pose[:3, :3])
        regions = detect_ghost_regions(
            clouds,
            poses,
            BalmParams(root_voxel_size=1.0, downsample_leaf=0.0),
            min_layer_keyframes=2,
            min_region_voxels=1,
        )
        self.assertEqual(regions, [], "opposite faces of a real wall were called a ghost")

    def test_ghost_merge_requires_spatial_connectivity(self) -> None:
        def region(key, center):
            return {
                "voxel_key": list(key),
                "view_side": 1,
                "center_xyz": list(center),
                "separation_m": 0.12,
                "normal": [0.0, 1.0, 0.0],
                "point_count": 100,
                "asymmetric": False,
                "layer_overlap_ratio": 0.8,
                "layer_support_fraction": 0.4,
                "layer_a_keyframes": [100, 120],
                "layer_b_keyframes": [900, 920],
                "suggested_pair": [110, 910],
            }

        raw = [
            region((0, 0, 0), (0.5, 0.5, 0.5)),
            region((1, 0, 0), (1.5, 0.5, 0.5)),
            region((20, 0, 0), (20.5, 0.5, 0.5)),
            region((21, 0, 0), (21.5, 0.5, 0.5)),
        ]
        merged = _merge_connected_ghost_voxels(
            raw,
            max_kf_gap=20,
            min_region_voxels=2,
            normal_cos_min=0.965,
            separation_tolerance=0.06,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(sorted(item["voxel_count"] for item in merged), [2, 2])
        for item in merged:
            self.assertLess(
                max(key[0] for key in item["voxel_keys"])
                - min(key[0] for key in item["voxel_keys"]),
                3,
            )

    def test_ghost_merge_tolerates_one_occluded_voxel(self) -> None:
        def region(key, ranges, pair):
            return {
                "voxel_key": list(key),
                "view_side": 1,
                "center_xyz": [float(v) + 0.5 for v in key],
                "separation_m": 0.12,
                "normal": [0.0, 1.0, 0.0],
                "point_count": 100,
                "asymmetric": False,
                "layer_overlap_ratio": 0.9,
                "layer_support_fraction": 0.4,
                "layer_a_keyframes": list(ranges[0]),
                "layer_b_keyframes": list(ranges[1]),
                "suggested_pair": list(pair),
            }

        # The dominant frames differ substantially, but both patches come from
        # the same two traversal intervals.  Voxel x=1 is empty due to occlusion.
        raw = [
            region((0, 0, 0), ((100, 180), (900, 980)), (105, 905)),
            region((2, 0, 0), ((160, 220), (960, 1020)), (215, 1015)),
        ]
        merged = _merge_connected_ghost_voxels(
            raw,
            max_kf_gap=20,
            min_region_voxels=2,
            normal_cos_min=0.965,
            separation_tolerance=0.06,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["voxel_count"], 2)
        self.assertEqual(merged[0]["candidate_type"], "duplicated_surface_region")
        self.assertFalse(merged[0]["loop_constraint_ready"])

    def test_duplicated_surface_ranking_is_area_not_point_density(self) -> None:
        def region(key, point_count, overlap_area):
            return {
                "voxel_key": list(key),
                "view_side": 1,
                "center_xyz": [float(value) + 0.5 for value in key],
                "separation_m": 0.12,
                "normal": [0.0, 1.0, 0.0],
                "point_count": point_count,
                "overlap_area_m2": overlap_area,
                "asymmetric": False,
                "layer_overlap_ratio": 0.8,
                "layer_support_fraction": 0.4,
                "layer_a_keyframes": [100, 120],
                "layer_b_keyframes": [900, 920],
                "suggested_pair": [110, 910],
            }

        merged = _merge_connected_ghost_voxels(
            [
                region((0, 0, 0), point_count=1000, overlap_area=0.25),
                region((20, 0, 0), point_count=100, overlap_area=2.0),
            ],
            max_kf_gap=20,
            min_region_voxels=1,
            normal_cos_min=0.965,
            separation_tolerance=0.06,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["voxel_keys"], [[20, 0, 0]])
        self.assertAlmostEqual(merged[0]["overlap_area_m2"], 2.0)
        self.assertAlmostEqual(merged[0]["surface_displacement_m3"], 0.24)

    def test_ghost_merge_rejects_nearby_parallel_surfaces(self) -> None:
        def region(key, center):
            return {
                "voxel_key": list(key),
                "view_side": 1,
                "center_xyz": list(center),
                "separation_m": 0.12,
                "normal": [1.0, 0.0, 0.0],
                "point_count": 100,
                "asymmetric": False,
                "layer_overlap_ratio": 0.9,
                "layer_support_fraction": 0.4,
                "layer_a_keyframes": [100, 120],
                "layer_b_keyframes": [900, 920],
                "suggested_pair": [110, 910],
            }

        merged = _merge_connected_ghost_voxels(
            [
                region((0, 0, 0), (0.1, 0.5, 0.5)),
                region((1, 0, 0), (1.1, 0.5, 0.5)),
            ],
            max_kf_gap=20,
            min_region_voxels=2,
            normal_cos_min=0.965,
            separation_tolerance=0.06,
        )
        self.assertEqual(merged, [])

    def test_no_planes_returns_input_poses(self) -> None:
        rng = np.random.default_rng(3)
        clouds = [rng.uniform(-5.0, 5.0, (30, 3)) for _ in range(2)]
        poses = np.stack([np.eye(4), _make_pose(10.0, np.array([1.0, 0.0, 0.0]))])
        params = BalmParams(root_voxel_size=1.0, downsample_leaf=0.0)
        result = run_balm_refinement(clouds, poses, params)
        np.testing.assert_allclose(result.poses_world_sensor, poses, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
