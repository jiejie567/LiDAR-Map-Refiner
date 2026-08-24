"""Zero-touch map repair: Initial Loop Search -> Audited PGO -> Final Map Refinement.

Map-inconsistency detection is diagnostic-only: it neither proposes loops nor
accepts, rejects, or retracts initial-loop constraints. Historical experiment
artifacts remain readable, but their iterative GhostLoop execution modes are no
longer part of LiDAR Map Refiner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

GUI_DIR = Path(__file__).resolve().parents[2] / 'gui'
sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure import (  # noqa: E402
    OFFICE_DEFAULT_VARIANCE_R_RAD2,
    OFFICE_DEFAULT_VARIANCE_T,
    RegistrationConfig,
    RegistrationWorkspace,
    load_tum_trajectory,
    matrix_to_quat_xyzw,
)
from manual_loop_closure.scan_context_io import (  # noqa: E402
    _gravity_canonical_rotation,
    load_scan_context_config,
    load_scan_context_gravity,
)
from manual_loop_closure.seed_loops import (  # noqa: E402
    build_descriptor_and_prepared_stack,
    build_descriptor_stack,
    descriptor_env_setup,
    find_seed_candidates,
    ghost_badness,
    pair_yaw_peaks,
)
from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
    BalmParams,
    detect_ghost_regions,
    prepare_local_cloud,
)
from manual_loop_closure.python_optimizer.oriented_surface_factors import (  # noqa: E402
    OrientedFactorConfig,
    estimate_oriented_surface_factor,
    prepare_oriented_cloud,
)
from manual_loop_closure.python_optimizer.loop_weighting import (  # noqa: E402
    CORRELATION_CLUSTERING_METHOD,
)
from manual_loop_closure.repair_presets import (  # noqa: E402
    NEAR_DUPLICATE_KEYFRAMES,
    PRODUCTION_PGO_PROFILE,
    REPAIR_GATE,
    REPAIR_POLICY,
    repair_environment_preset,
    repair_gate,
)
from manual_loop_closure.concurrency import hypothesis_workers  # noqa: E402
from experiment_io import (  # noqa: E402
    CONSTRAINT_HEADER,
    experiment_controls_payload,
    load_initial_constraints_csv,
)
from causal_audit import (  # noqa: E402
    evaluate_causal_effect,
    evaluate_graph_trial,
    factor_consistency,
    freeze_region_evidence,
    gravity_consistency,
    measure_frozen_evidence,
)
from proposal_ledger import ProposalLedger, proposal_id  # noqa: E402

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    'session', nargs='?', type=Path,
    default=Path('/home/anyverse/icra2027_runtime/manual_loop/original_demo_auto'))
_parser.add_argument(
    'mode', nargs='?', default='production',
    choices=('production',),
    help=(
        'production runs Initial Loop Search -> Audited PGO -> one Final Map '
        'Refinement (double-sided BALM).'
    ),
)
_parser.add_argument('environment', nargs='?', default='indoor')
_parser.add_argument(
    '--balm-root-voxel-size', '--balm_root_voxel_size', type=float,
    help='Experiment-level BALM association and diagnostic root voxel size.')
_parser.add_argument(
    '--initial-constraints-csv', '--initial_constraints_csv', type=Path,
    help=(
        'Frozen enabled initial-loop checkpoint; skips descriptor retrieval '
        'and initial-loop GICP.'
    ))
_parser.add_argument(
    '--result-json', type=Path,
    help=(
        'Write an atomic machine-readable success contract for callers such '
        'as the GUI. The file is created only after the complete terminal '
        'output has been materialized.'
    ),
)
_parser.add_argument(
    '--diagnose-map-inconsistency', action='store_true',
    help=(
        'Run the optional inspection-only map-inconsistency scan during the '
        'terminal refinement. It never proposes or audits loop constraints.'
    ),
)
_parser.add_argument(
    '--loop-factor-mode',
    choices=('diagonal', 'oriented_anisotropic'),
    default='diagonal',
    help=(
        'Loop measurement/noise construction. Production keeps the accepted '
        'GICP measurement with the cross-dataset diagonal profile; oriented '
        'remeasurement remains an explicit ablation.'
    ),
)
_parser.add_argument(
    '--factor-information-scale', type=float,
    default=0.00016358831377740368,
    help='Frozen cross-dataset calibration scale for oriented factor Hessians.',
)
_parser.add_argument(
    '--gravity-max-error-deg', type=float,
    default=float(PRODUCTION_PGO_PROFILE['gravity_sanity_limit_deg']),
    help=(
        'Maximum angle between the registered source up-vector and target '
        'up-vector. Production uses only a catastrophic sanity limit; gravity '
        'is not an accuracy gate.'
    ),
)
_parser.add_argument(
    '--pgo-profile', choices=('legacy', 'calibrated'), default='calibrated',
    help=(
        'legacy reproduces direct iSAM2 with unscaled independent loop factors; '
        'calibrated uses loop-only GNC-TLS and the frozen conservative '
        'translation/rotation information profile.'
    ),
)
_parser.add_argument(
    '--pgo-information-scale', type=float,
    help='Override the selected profile global scale for added loop factors.',
)
_parser.add_argument(
    '--pgo-correlation-window-keyframes', type=int,
    help='Override the selected profile endpoint-neighbourhood window.',
)
_parser.add_argument(
    '--pgo-cluster-information-budget', type=float,
    help='Override the selected profile effective-factor budget per cluster.',
)
_parser.add_argument(
    '--pgo-correlation-policy', choices=('pair', 'shared_endpoint'),
    help='Override whether correlation requires both endpoints or either endpoint.',
)
_parser.add_argument(
    '--pgo-cluster-allocation', choices=('equal', 'representative'),
    help='Override how a correlated cluster distributes its information budget.',
)
ARGS = _parser.parse_args()
SESSION = ARGS.session.expanduser()
RESULT_JSON = (
    ARGS.result_json.expanduser().resolve()
    if ARGS.result_json is not None else None
)
MODE = ARGS.mode  # production only
ENV = ARGS.environment  # indoor|outdoor
LOOP_FACTOR_MODE = ARGS.loop_factor_mode
FACTOR_INFORMATION_SCALE = float(ARGS.factor_information_scale)
if ENV not in {'indoor', 'outdoor'}:
    raise SystemExit(f'unsupported environment preset: {ENV}')
INPUT_SHA256 = {
    'optimized_tum': hashlib.sha256(
        (SESSION / 'optimized_poses_tum.txt').read_bytes()
    ).hexdigest(),
    'pose_graph_g2o': hashlib.sha256(
        (SESSION / 'pose_graph.g2o').read_bytes()
    ).hexdigest(),
}
if not np.isfinite(FACTOR_INFORMATION_SCALE) or FACTOR_INFORMATION_SCALE <= 0.0:
    raise SystemExit('--factor-information-scale must be finite and positive')
if ARGS.pgo_profile == 'legacy':
    PGO_OPTIMIZE_MODE = 'isam2'
    PGO_INFORMATION_SCALE = 1.0
    PGO_CORRELATION_WINDOW = 0
    PGO_CLUSTER_BUDGET = 0.0
    PGO_CORRELATION_POLICY = 'pair'
    PGO_CLUSTER_ALLOCATION = 'equal'
else:
    PGO_OPTIMIZE_MODE = str(PRODUCTION_PGO_PROFILE['optimize_mode'])
    # Frozen four-sequence development selection, checked once without
    # retuning on M2DGR hall02/hall04.  The holdout delta was +1 mm / 0 mm.
    PGO_INFORMATION_SCALE = float(
        PRODUCTION_PGO_PROFILE['manual_information_scale']
    )
    # A deterministic complete-link budget fixed the earlier transitive-chain
    # bug, but the frozen four-dataset prefix ablation still favored no budget
    # (geometric-mean final ATE ratio 0.6564 vs 0.6581).  Keep the scalar
    # approximation off until a joint covariance/shared-bias factor earns its
    # added complexity.
    PGO_CORRELATION_WINDOW = int(
        PRODUCTION_PGO_PROFILE['correlation_window_keyframes']
    )
    PGO_CLUSTER_BUDGET = float(
        PRODUCTION_PGO_PROFILE['cluster_information_budget']
    )
    PGO_CORRELATION_POLICY = str(
        PRODUCTION_PGO_PROFILE['correlation_policy']
    )
    PGO_CLUSTER_ALLOCATION = str(
        PRODUCTION_PGO_PROFILE['cluster_allocation']
    )
if ARGS.pgo_information_scale is not None:
    PGO_INFORMATION_SCALE = float(ARGS.pgo_information_scale)
if ARGS.pgo_correlation_window_keyframes is not None:
    PGO_CORRELATION_WINDOW = int(ARGS.pgo_correlation_window_keyframes)
if ARGS.pgo_cluster_information_budget is not None:
    PGO_CLUSTER_BUDGET = float(ARGS.pgo_cluster_information_budget)
if ARGS.pgo_correlation_policy is not None:
    PGO_CORRELATION_POLICY = ARGS.pgo_correlation_policy
if ARGS.pgo_cluster_allocation is not None:
    PGO_CLUSTER_ALLOCATION = ARGS.pgo_cluster_allocation
if not np.isfinite(PGO_INFORMATION_SCALE) or PGO_INFORMATION_SCALE <= 0.0:
    raise SystemExit('--pgo-information-scale must be finite and positive')
if PGO_CORRELATION_WINDOW < 0:
    raise SystemExit('--pgo-correlation-window-keyframes must be non-negative')
if not np.isfinite(PGO_CLUSTER_BUDGET) or PGO_CLUSTER_BUDGET < 0.0:
    raise SystemExit('--pgo-cluster-information-budget must be non-negative')
STATS = {'proposed': 0, 'gicp_pairs': 0, 'accepted': 0}
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LEDGER_PATH = SESSION / 'proposal_ledgers' / f'{RUN_ID}.jsonl'
LEDGER = ProposalLedger(LEDGER_PATH, RUN_ID)
PY = sys.executable
CLI_DIR = GUI_DIR / 'manual_loop_closure' / 'python_optimizer'
# Two independent criteria, matching the C++ relocalisation path this tool was
# ported from (prior_icp.cpp): how much of the scan found a partner at all
# (overlap, larger is better) and how far those partners sit (residual, smaller
# is better). The orientation test is NOT folded into either -- it decides
# ghost-versus-thin-wall, which is a different question from whether a
# registration is correct. Overlap 0.5 is the value already validated in the
# C++ relocaliser; the residual bound is 0.9x the voxel edge, since voxel
# quantisation sets the floor a correct registration cannot beat.
# Both thresholds come from repair_presets.repair_gate so this evaluator and
# the GUI cannot drift apart; the values are resolved below, once the
# environment preset has told us the voxel edge.
MIN_FITNESS = MAX_RMSE = None
MAX_ROUNDS = int(os.environ.get('GHOSTLOOP_MAX_ROUNDS', '20'))
# 20, not the 10 this was. 10 was chosen as an oscillation backstop nothing was
# expected to reach, and on eight of nine sequences nothing does -- they settle in
# one to seven rounds. MCD ntu_day_01 does reach it, and measuring past it showed
# the cap was the binding constraint rather than a safety net: with room to run it
# converges at round 12 on 36 constraints and 224.4 cm, against 35 constraints and
# 234.5 cm when stopped at 10. A backstop that changes the answer is not a backstop.
# 20 keeps a real guard (no sequence has come close) without truncating convergence.
# 轮次初值集：identity + 单峰 SC-yaw + 单符号分离量（加在前两个假设上）= 4 个。
# 曾经是 11 个：多峰 SC-yaw、+-两个符号的分离量、每个峰一个共位。三处都砍掉了，
# 2026-08-05 在 FusionPortable escalator00（室内、原生缺陷、有真值）实测：
#     11 初值 + 四档级联   16.8 cm   5 条约束
#      4 初值 + 四档级联   16.8 cm   5 条约束      <- 逐位相同
# 胜出的初值与档位也一字未变（21->67 仍是 annealed seed 1，4->121 仍是 annealed
# seed 0），说明砍掉的那些本来就没赢过。配准次数降到 2.75 分之一。
#
# 三条各自的理由：
#  * 共位是把源直接搬到目标处，为初始回环搜索设计——候选来自漂移地图上的地点识别，
#    两端可能相隔几十米。轮次提案来自已经基本正确的地图上诊断出的鬼影，两层只差
#    8-35 cm，共位只会把源扔到地图另一头。统计也一致：49 条轮次约束里共位赢 0 条。
#  * 多峰 SC-yaw 是为对称场景准备的，同样属于"不知道自己在哪"的情形。
#  * 分离量的符号本来就是确定的，见下面 sep_sign 处的说明——之前两个符号都试，
#    是代码把层次归属丢了，不是几何上的不确定。
#
# 注意这和 2026-08-04 测过的"砍成 identity+-分离量"不是一回事：那次连 SC-yaw
# 一起砍了，而分离量是加在 deltas[:2]（identity 和第一个 SC-yaw）上的，所以
# "SC-yaw 附近的分离量"也一并没了，ntu 从 23 条掉到 4 条。
#
# 也不要把这里的级联换成体素由粗到细。实测过：初值一个不动、只把四档换成
# 0.25->0.20 两档，escalator00 就从 5 条 16.8 cm 掉到 2 条 22.0 cm。粗体素扩大的
# 是"大致对但偏一点"的收敛半径，而 wide 档扩的是对应半径、submap 档补的是源侧
# 几何、anneal 档做的是"先用 wide 热启动再用 near 收紧"的两段式——粗体素只覆盖第一种。
# （anneal 原本的定义是"关掉法向门做热启动"；法向门 2026-08-06 移除后它退化成只差
#   对应半径与迭代数的两段式，见 545 行附近的说明。）
COLLOC_IN_ROUNDS = os.environ.get('GHOSTLOOP_COLLOC_ROUNDS', '0') != '0'
# 由粗到细替代多假设级联。多初值和多分辨率是同一个问题的两条解法——都是为了让
# ICP 不掉进错误的局部最优——没必要同时用。粗体素把两片点云糊成大结构，收敛盆地
# 宽；解出来的位姿当下一档更细体素的初值，逐级找回精度。
# 空字符串表示沿用现有的四档多假设级联。
C2F = [float(v) for v in os.environ.get('GHOSTLOOP_C2F', '').split(',') if v.strip()]
# 轮次是否只取描述子相关曲线的最高峰。多峰是为对称场景准备的，但在鬼影轮次上
# 地图已经基本正确，次高峰的价值存疑。
ROUND_SINGLE_PEAK = os.environ.get('GHOSTLOOP_SINGLE_PEAK', '1') == '1'
SEP_PRIOR = os.environ.get('GHOSTLOOP_SEP_PRIOR', '1') != '0'
# 只给检测器指出的那一个符号，而不是两个都试。
SEP_SIGNED = os.environ.get('GHOSTLOOP_SEP_SIGNED', '1') == '1'
HYPOTHESIS_WORKERS = hypothesis_workers()
# The loop factor's noise model, taken from the same constants the GUI writes
# (manual_loop_closure_tool.ManualConstraint.csv_row) so the two entry points
# cannot disagree. This used to be a hardcoded sqrt(0.1); when the translational
# variance was recalibrated 0.1 -> 1.6 on 2026-08-06 the GUI picked it up and
# this file did not, so every headless run solved at a loop/odometry sigma ratio
# of 10 while the GUI ran at 40 -- measured on the in-house session, the same
# 18 constraints give trajectories 4.4 cm apart at the median, 17.0 cm at worst.
# Translational and rotational are kept separate because they are not the same
# number and only looked alike while both were sqrt(0.1).
#
# The legacy profile preserves the historical radian-variance-to-degree-column
# quirk for exact reproduction.  The calibrated profile writes the explicit
# degree sigma from PRODUCTION_PGO_PROFILE and must not inherit that unit bug.
SIGMA_T = f'{math.sqrt(OFFICE_DEFAULT_VARIANCE_T[0]):.6f}'
SIGMA_R = (
    f"{float(PRODUCTION_PGO_PROFILE['rotation_sigma_deg']):.6f}"
    if ARGS.pgo_profile == 'calibrated'
    else f'{math.sqrt(OFFICE_DEFAULT_VARIANCE_R_RAD2[0]):.6f}'
)

KF_DIR = SESSION / 'key_point_frame'
config, CHANNEL_WEIGHTS = descriptor_env_setup(load_scan_context_config(SESSION), ENV)
trajectory = load_tum_trajectory(SESSION / 'optimized_poses_tum.txt')
workspace = RegistrationWorkspace(KF_DIR, trajectory)
repair_preset = repair_environment_preset(ENV)
if ARGS.balm_root_voxel_size is not None and ARGS.balm_root_voxel_size <= 0.0:
    raise SystemExit('--balm-root-voxel-size must be positive')
BALM_ROOT_VOXEL_SIZE = (
    float(ARGS.balm_root_voxel_size)
    if ARGS.balm_root_voxel_size is not None
    else float(repair_preset['balm_voxel'])
)
INITIAL_CONSTRAINTS_CSV = (
    ARGS.initial_constraints_csv.expanduser().resolve()
    if ARGS.initial_constraints_csv is not None else None
)
_gate = repair_gate(repair_preset['voxel'])
MIN_FITNESS, MAX_RMSE = _gate['min_fitness'], _gate['max_rmse']
# Benchmark-only relaxation of the admission gate. Not a product switch and not
# read by the GUI: it exists so the paper can show what the POST-admission
# safeguards do, which is invisible while the gate is tight enough that no false
# loop ever gets in. Deliberately loosening it lets the aliased in-house
# corridors produce false constraints on purpose, and the odometry budget, the
# temporal filter, probation and the post-optimization consistency check are
# then measured on what they actually catch. Unset, both keep the shipped values.
_relax_f = os.environ.get('GHOSTLOOP_GATE_FITNESS')
_relax_r = os.environ.get('GHOSTLOOP_GATE_RMSE')
if _relax_f:
    MIN_FITNESS = float(_relax_f)
if _relax_r:
    MAX_RMSE = float(_relax_r)
if ENV == 'outdoor':
    reg_config = RegistrationConfig(max_correspondence_distance=repair_preset['max_corr'],
                                    voxel_size=repair_preset['voxel'],
                                    target_map_voxel_size=repair_preset['target_map_voxel'],
                                    target_neighbors=repair_preset['target_neighbors'])
    wide_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                     voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                     target_neighbors=repair_preset['target_neighbors'])
    submap_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                       voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                       target_neighbors=repair_preset['target_neighbors'],
                                       source_window=10)
    anneal_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                       voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                       target_neighbors=repair_preset['target_neighbors'])
    balm_params = BalmParams(
        root_voxel_size=BALM_ROOT_VOXEL_SIZE,
        downsample_leaf=repair_preset['balm_downsample'],
        max_range=repair_preset['balm_max_range'],
        max_iterations=repair_preset['balm_iterations'],
        double_sided_enable=repair_preset['balm_double_sided'])
    OBS_RANGE = repair_preset['observation_range']
    # Outdoor scans at 0.4 m voxels carry more surface roughness; GT-checked on
    # Sejong01: a correct loop registers at RMSE ~0.22 (pose error 20 cm/0.4 deg)
    # while aliased pairs sit at 0.55+, so 0.30 separates them cleanly.
    # GT-checked on kth_night_01 with the criteria separated: the two genuine
    # revisits reach overlap 1.00 at residual 0.249/0.250, three descriptor
    # aliases 136-226 m away also overlap well (0.76-0.88) but sit at residual
    # 0.722-0.885. Overlap alone cannot reject the aliases and residual alone
    # cannot confirm a sparse match; together they separate cleanly.
    # MAX_RMSE now comes from repair_gate(0.4) = 0.36.
else:
    reg_config = RegistrationConfig(
        max_correspondence_distance=repair_preset['max_corr'],
        voxel_size=repair_preset['voxel'],
        target_map_voxel_size=repair_preset['target_map_voxel'],
        target_neighbors=repair_preset['target_neighbors'])
    wide_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                     voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                     target_neighbors=repair_preset['target_neighbors'])
    submap_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                       voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                       target_neighbors=repair_preset['target_neighbors'],
                                       source_window=10)
    anneal_config = RegistrationConfig(max_correspondence_distance=repair_preset['wide_max_corr'], max_iterations=100,
                                       voxel_size=repair_preset['voxel'], target_map_voxel_size=repair_preset['target_map_voxel'],
                                       target_neighbors=repair_preset['target_neighbors'])
    balm_params = BalmParams(
        root_voxel_size=BALM_ROOT_VOXEL_SIZE,
        downsample_leaf=repair_preset['balm_downsample'],
        max_range=repair_preset['balm_max_range'],
        max_iterations=repair_preset['balm_iterations'],
        double_sided_enable=repair_preset['balm_double_sided'])
    OBS_RANGE = repair_preset['observation_range']

# OBS_RANGE trims the clouds the Scan Context stack is built from; the ghost
# diagnosis uses its own, tighter number. See repair_presets for why they differ.
GHOST_RANGE = REPAIR_POLICY['ghost_observation_range_m']
diagnosis_params = replace(
    balm_params,
    max_range=min(balm_params.max_range, OBS_RANGE),
)
ORIENTED_FACTOR_CONFIG = OrientedFactorConfig(
    voxel_size_m=0.20,
    normal_radius_m=0.60,
    max_correspondence_m=0.60,
    normal_gate_deg=45.0,
    min_overlap=0.20,
    min_correspondences=80,
    information_scale=FACTOR_INFORMATION_SCALE,
)
_ORIENTED_CLOUD_CACHE = {}
_FACTOR_RESULT_CACHE = {}
_LAST_FACTOR_REASON = {}

def log(msg):
    print(msg, flush=True)

def _information_upper(matrix):
    return [
        float(matrix[row, col])
        for row in range(6) for col in range(row, 6)
    ]


def _information_from_constraint_row(row):
    if len(row) > len(CONSTRAINT_HEADER) and row[len(CONSTRAINT_HEADER)]:
        values = json.loads(row[len(CONSTRAINT_HEADER)])
        if len(values) != 21:
            raise ValueError('full factor information must contain 21 values')
        information = np.zeros((6, 6), dtype=float)
        cursor = 0
        for matrix_row in range(6):
            for matrix_col in range(matrix_row, 6):
                information[matrix_row, matrix_col] = values[cursor]
                information[matrix_col, matrix_row] = values[cursor]
                cursor += 1
        return information
    sigma_t = np.asarray(row[10:13], dtype=float)
    # Preserve the optimizer's historical CSV convention exactly: the three
    # rotation values are interpreted as degrees by optimizer.py.
    sigma_r = np.deg2rad(np.asarray(row[13:16], dtype=float))
    return np.diag(1.0 / np.square(np.concatenate([sigma_t, sigma_r])))


def factor_residual_details(tum_path, rows):
    """Return factor residuals in both physical and uncertainty-normalized units."""
    T = load_tum_trajectory(tum_path).transforms_world_sensor
    out = []
    for idx, row in enumerate(rows):
        if row[0] != '1':
            out.append({
                'index': idx, 'translation_m': 0.0, 'rotation_deg': 0.0,
                'nis': 0.0, 'normalized_score': 0.0, 'gate_value': 0.0,
            })
            continue
        s_id, t_id = int(row[1]), int(row[2])
        meas = np.eye(4)
        meas[:3, :3] = quat_xyzw_to_matrix(
            [float(row[6]), float(row[7]), float(row[8]), float(row[9])])
        meas[:3, 3] = [float(row[3]), float(row[4]), float(row[5])]
        solved = np.linalg.inv(T[t_id]) @ T[s_id]
        consistency = factor_consistency(
            meas, solved,
            _information_from_constraint_row(row) * PGO_INFORMATION_SCALE,
        )
        out.append({
            'index': idx,
            'translation_m': consistency.translation_residual_m,
            'rotation_deg': consistency.rotation_residual_deg,
            'nis': consistency.nis,
            'chi_square_threshold': consistency.chi_square_threshold,
            'normalized_score': consistency.normalized_score,
            'gate_value': consistency.normalized_score,
        })
    return out


def factor_residuals(tum_path, rows):
    """Compatibility view used by leave-one-out: (gate value, row index)."""
    return [
        (item['gate_value'], item['index'])
        for item in factor_residual_details(tum_path, rows)
    ]


def quat_xyzw_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_LAST_BALM_MOTION_STATS = {}


def balm_sane(
    pgo_tum,
    balm_tum,
    limit=REPAIR_POLICY['balm_sanity_limit_m'],
    rotation_limit_deg=REPAIR_POLICY['balm_sanity_limit_rotation_deg'],
):
    """BALM sanity gate: with a fixed constraint set, plane refinement should
    move poses centimeters. Meter-scale motion means the BA geometry is
    degenerate (e.g. no horizontal support in narrow-FOV data) and its output
    must be rejected in favor of the PGO poses."""
    global _LAST_BALM_MOTION_STATS
    _LAST_BALM_MOTION_STATS = {}
    try:
        before = load_tum_trajectory(pgo_tum)
        after = load_tum_trajectory(balm_tum)
    except Exception as exc:
        log(f'[balm-gate] cannot validate BALM trajectory ({exc}) -> '
            'rejecting BA output, keeping PGO poses')
        return False
    a, b = before.positions_xyz, after.positions_xyz
    ta = before.transforms_world_sensor
    tb = after.transforms_world_sensor
    if (
        len(a) == 0 or len(a) != len(b)
        or not np.isfinite(ta).all() or not np.isfinite(tb).all()
    ):
        log(f'[balm-gate] invalid BALM trajectory shape/values '
            f'({len(a)} input vs {len(b)} output) -> rejecting BA output, '
            'keeping PGO poses')
        return False
    mean_delta = float(np.linalg.norm(a - b, axis=1).mean())
    relative_rotation = np.einsum(
        'nji,njk->nik', ta[:, :3, :3], tb[:, :3, :3]
    )
    mean_rotation_deg = float(np.degrees(
        Rotation.from_matrix(relative_rotation).magnitude()
    ).mean())
    _LAST_BALM_MOTION_STATS = {
        'measured_mean_pose_motion_m': mean_delta,
        'measured_mean_pose_rotation_deg': mean_rotation_deg,
    }
    if mean_delta > limit:
        log(f'[balm-gate] refinement moved poses {mean_delta:.2f} m on average '
            f'(> {limit}) -> rejecting BA output, keeping PGO poses')
        return False
    if mean_rotation_deg > rotation_limit_deg:
        log(f'[balm-gate] refinement rotated poses {mean_rotation_deg:.2f}° '
            f'on average (> {rotation_limit_deg}°) -> rejecting BA output, '
            'keeping PGO poses')
        return False
    return True


def run_balm_cli(pgo_out, suffix):
    """Run exactly one coarse-to-fine BALM pass from a completed PGO run."""
    balm_out = pgo_out.parent / (pgo_out.name + suffix)
    balm_args = [
        PY, '-u', str(CLI_DIR / 'balm_cli.py'),
        '--tum', str(pgo_out / 'optimized_poses_tum.txt'),
        '--keyframe-dir', str(KF_DIR), '--output-dir', str(balm_out),
        '--source-run-dir', str(pgo_out),
        '--max-iterations', str(balm_params.max_iterations),
        '--root-voxel', str(balm_params.root_voxel_size),
        '--downsample-leaf', str(balm_params.downsample_leaf),
        '--max-range', str(balm_params.max_range),
        '--max-observation-range', str(GHOST_RANGE),
        ('--double-sided' if balm_params.double_sided_enable
         else '--no-double-sided'),
    ]
    # Production ends after map refinement. The historical research modes
    # still need the optional map-inconsistency ledger for their ablations.
    if ARGS.diagnose_map_inconsistency:
        balm_args.append('--diagnose-map-inconsistency')
    completed = subprocess.run(
        balm_args,
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        log(completed.stdout[-1500:])
        log(completed.stderr[-1500:])
    return completed, balm_out


def adopt_balm_or_restore_pgo(pgo_out, balm_out):
    """Adopt a bounded BALM result; otherwise keep a graph/pose-consistent PGO."""
    adopted = balm_sane(
        pgo_out / 'optimized_poses_tum.txt',
        balm_out / 'optimized_poses_tum.txt',
    )
    if not adopted:
        for name in ('optimized_poses_tum.txt', 'pose_graph.g2o'):
            source = pgo_out / name
            if source.is_file():
                shutil.copy2(source, balm_out / name)
    report_path = balm_out / 'balm_report.json'
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        report = {}
    report['production_adoption'] = {
        'adopted': bool(adopted),
        'sanity_limit_mean_pose_motion_m': float(
            REPAIR_POLICY['balm_sanity_limit_m']),
        'sanity_limit_mean_pose_rotation_deg': float(
            REPAIR_POLICY['balm_sanity_limit_rotation_deg']),
        **_LAST_BALM_MOTION_STATS,
        'fallback': None if adopted else 'restored_pgo_trajectory_and_graph',
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return adopted

# NOTE on gate thresholds: adjacent-pair calibration was tried and measured a
# ~1 cm session noise floor, i.e. revisit-pair RMSE (0.15-0.22 on true loops)
# is dominated by viewpoint/scene change between visits, not sensor noise —
# a flat per-environment threshold with the budget/temporal/probation
# safeguards behind it is the honest design. Pairs that fail are surfaced to
# the operator (human-in-the-loop fallback), e.g. V1's start-end revisit where
# the scene physically changed between visits.

# Odometry-consistency budget: a loop's correction cannot exceed what the
# odometry could plausibly have drifted over the chain between the pair.
# Self-similar streets register confidently at the wrong offset (fitness and
# RMSE both look fine); only the odometry chain exposes the contradiction.
_ODO_T0 = trajectory.transforms_world_sensor.copy()
_ODO_CUM = np.concatenate([[0.0], np.cumsum(
    np.linalg.norm(np.diff(trajectory.positions_xyz, axis=0), axis=1))])
BUDGET_BASE = REPAIR_POLICY['budget_base_outdoor_m' if ENV == 'outdoor'
                            else 'budget_base_indoor_m']
# 基数与比率的取值理由见 repair_presets.REPAIR_POLICY（2026-08-06 从两处字面量收拢）。
BUDGET_RATE = float(os.environ.get('GHOSTLOOP_BUDGET_RATE',
                                   REPAIR_POLICY['budget_rate']))
# A constraint that survives optimization disagreeing with the graph by more
# than this is contradicting the other constraints, not merely imprecise:
# GT-verified sets settle below 0.10 m while a single 3 m error pushes two
# factors past 1.5 m.
# Both surviving criteria are derived from quantities the pipeline already
# has, not fitted to any sequence:
#   * a correct registration cannot beat the voxel quantisation its own
#     downsampling imposes, and a wrong one sits at scene scale, so the RMSE
#     bound is a multiple of the voxel edge;
#   * a constraint is inconsistent when the graph cannot satisfy it within its
#     own stated uncertainty, i.e. beyond 3 sigma of the factor's noise model.
# Reported in the run summary only. It used to be a second, independent copy of
# the gate ratio that had drifted to 0.9 while repair_presets.REPAIR_GATE moved to
# 1.1, so every auto_repair_summary.json advertised a bound the run did not use.
# Read it from the one source of truth instead of restating it.
RMSE_VOXEL_RATIO = REPAIR_GATE['rmse_voxel_ratio']
RESIDUAL_LIMIT = REPAIR_POLICY['residual_limit_m']
CONSISTENCY_LIMIT = 1.0
# Occam check: with GHOSTLOOP_MINIMAL=1 the pipeline keeps only the two
# criteria the evidence actually supports -- per-pair RMSE normalised by voxel
# size, and post-optimization factor consistency -- and drops fitness, the
# odometry budget, the temporal filter and probation, each of which was added
# to fix one observed failure and may be subsumed by the other two.
MINIMAL = os.environ.get('GHOSTLOOP_MINIMAL') == '1'
# Benchmark-only: cap how many constraints one round may admit. The shipped
# pipeline takes everything that passes, and on every sequence the first
# round then does nearly all the work. Rate-limiting admission asks whether
# the fixed point depends on that eagerness or only on the constraint set the
# map eventually supports -- i.e. whether the cycle is order-invariant.
PER_ROUND_CAP = int(os.environ.get('GHOSTLOOP_PER_ROUND_CAP', '0'))
# Benchmark-only: cap Initial Loop Search alone, leaving legacy rounds unrestricted. This
# separates 'how much appearance evidence is available to start' from 'how
# fast admission proceeds', which the single cap above conflates. Setting it
# to 1 asks the sharpest version of the question: is one revisit enough to
# bring the map within the detector's reach, after which it supplies the rest?
SEED_CAP = int(os.environ.get('GHOSTLOOP_SEED_CAP', '0'))
# Benchmark-only: keep only the widest-spanning seed and discard the rest.
# The point of this one is different from a plain cap. Retrieval's unique
# contribution is the loop that fixes the trajectory's global shape -- the
# one whose two ends are thousands of keyframes apart -- and no amount of
# local diagnosis substitutes for it. Keeping exactly that seed asks whether
# everything else can then be recovered from the map alone.
# The value is a rank: 1 keeps the widest seed, 2 the second widest, and so
# on, which is what distinguishes 'the spanning loop matters' from 'any long
# loop will do'.
SEED_WIDEST = int(os.environ.get('GHOSTLOOP_SEED_WIDEST', '0'))
# Benchmark-only: turn the post-optimization consistency retraction off so the
# relaxed-gate ablation can report the ATE with and without it on the same
# admitted constraint set. Paired with GHOSTLOOP_GATE_* above.
NO_CONSISTENCY = os.environ.get('GHOSTLOOP_NO_CONSISTENCY') == '1'
STAGE0_AUDIT_ONLY = (
    os.environ.get('GHOSTLOOP_INITIAL_LOOPS_AUDIT_ONLY') == '1'
    or os.environ.get('GHOSTLOOP_STAGE0_AUDIT_ONLY') == '1'
)
GRAVITY_MAX_ERROR_DEG = float(ARGS.gravity_max_error_deg)
if not np.isfinite(GRAVITY_MAX_ERROR_DEG) or GRAVITY_MAX_ERROR_DEG <= 0.0:
    raise SystemExit('--gravity-max-error-deg must be finite and positive')
# See repair_presets.NEAR_DUPLICATE_KEYFRAMES for what this window is and why
# it is 30; it is shared with the GUI so the two cannot disagree.
NEAR_DUP = NEAR_DUPLICATE_KEYFRAMES
if MINIMAL:
    MIN_FITNESS = 0.0
    BUDGET_RATE = 1e9

REPAIR_SUMMARY = SESSION / 'auto_repair_summary.json'
INITIAL_CONSTRAINT_COUNT = 0


def write_repair_summary(status, **extra):
    payload = {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'status': status,
        'session': str(SESSION.resolve()),
        'mode': MODE,
        'environment': ENV,
        'workflow': {
            'phases': [
                'initial_loop_search',
                'audited_pgo',
                'final_map_refinement',
            ],
            'diagnostics': 'optional_map_inconsistency_scan',
            'legacy_schema_aliases': {
                'stage0': 'initial_loop_search',
                'seed': 'initial_loop_search',
                'auto_seed': 'initial_loop_search',
            },
        },
        'descriptor': {
            'dual_z_layer_enable': bool(config.dual_z_layer_enable),
            'gravity_canonicalization_enable': bool(
                config.gravity_canonicalization_enable),
            # Single-channel descriptors intentionally use ``None`` (for
            # outdoor data and indoor sessions whose recorded Scan Context
            # config has dual-z disabled).  Preserve that semantic value in
            # the run manifest instead of treating it as an iterable.
            'channel_weights': (
                None if CHANNEL_WEIGHTS is None
                else [float(value) for value in CHANNEL_WEIGHTS]
            ),
            'min_joint_rings': int(config.min_joint_rings),
            'retrieval_height_offset': float(
                config.retrieval_height_offset),
            'sector_support_exponent': float(
                config.sector_support_exponent),
        },
        'stage0_retrieval': {
            'candidate_pool': int(globals().get(
                '_SEED_N', repair_preset['seed_max_candidates']
            )),
            'max_descriptor_distance': float(globals().get(
                '_SEED_D', repair_preset['seed_max_distance']
            )),
            'segment_radius_keyframes': int(globals().get(
                '_SEED_SEGMENT_RADIUS',
                min(100, max(10, trajectory.size // 8)),
            )),
            'ring_key_top_k': int(globals().get('_SEED_RING_KEY_TOP_K', 10)),
        },
        'registration': {
            'near': asdict(reg_config),
            'wide': asdict(wide_config),
            'submap': asdict(submap_config),
            'anneal': asdict(anneal_config),
        },
        'loop_factor': {
            'mode': LOOP_FACTOR_MODE,
            'measurement_convention': 'target_to_source',
            'information_order': 'g2o_translation_rotation',
            'config': (
                {
                    'translation_sigma_m': float(SIGMA_T),
                    'rotation_sigma_deg': float(SIGMA_R),
                }
                if LOOP_FACTOR_MODE == 'diagonal'
                else asdict(ORIENTED_FACTOR_CONFIG)
            ),
            'uncertainty_calibration': (
                'diagonal cross-dataset production profile'
                if LOOP_FACTOR_MODE == 'diagonal' else
                'cross-dataset oriented Hessian scale'
            ),
        },
        'pgo': {
            'profile': ARGS.pgo_profile,
            'optimize_mode': PGO_OPTIMIZE_MODE,
            'manual_information_scale': PGO_INFORMATION_SCALE,
            'correlation_window_keyframes': PGO_CORRELATION_WINDOW,
            'cluster_information_budget': PGO_CLUSTER_BUDGET,
            'correlation_policy': PGO_CORRELATION_POLICY,
            'correlation_clustering': CORRELATION_CLUSTERING_METHOD,
            'cluster_allocation': PGO_CLUSTER_ALLOCATION,
            'base_graph_treated_as_known_inliers': (
                PGO_OPTIMIZE_MODE == 'gnc_tls'
            ),
        },
        'balm': asdict(balm_params),
        'experiment_controls': experiment_controls_payload(
            balm_params.root_voxel_size,
            INITIAL_CONSTRAINTS_CSV,
            INITIAL_CONSTRAINT_COUNT,
        ),
        'gates': {
            'min_overlap': MIN_FITNESS,
            'max_rmse_m': MAX_RMSE,
            'rmse_voxel_ratio': RMSE_VOXEL_RATIO,
            'residual_limit_m': RESIDUAL_LIMIT,
            'factor_consistency_metric': (
                'scaled NIS / chi2_0.9973(df=6)'
            ),
            'factor_consistency_limit': CONSISTENCY_LIMIT,
            'gravity_max_error_deg': GRAVITY_MAX_ERROR_DEG,
            'budget_base_m': BUDGET_BASE,
            'budget_rate': BUDGET_RATE,
            'near_duplicate_keyframes': NEAR_DUP,
            'ghost_observation_range_m': GHOST_RANGE,
            'descriptor_trim_range_m': OBS_RANGE,
        },
        'statistics': dict(STATS),
        'proposal_ledger': {
            'path': str(LEDGER_PATH.resolve()),
            'run_id': RUN_ID,
            'schema_version': 1,
        },
        'causal_audit': {
            'fixed_evidence_close_threshold_m': 0.08,
            'fixed_evidence_min_absolute_drop_m': 0.03,
            'fixed_evidence_min_relative_drop': 0.30,
            'balm_in_evidence_loop': False,
        },
        'pipeline': {
            'production_default': MODE == 'production',
            'map_inconsistency_role': (
                'diagnostic_only' if MODE == 'production'
                else 'legacy_proposal_channel'
            ),
            'balm_policy': (
                'one_final_double_sided_pass' if MODE == 'production'
                else 'legacy_mode_specific'
            ),
        },
        'stage0_audit_only': STAGE0_AUDIT_ONLY,
        **extra,
    }
    # Preferred report name. Keep ``stage0_retrieval`` byte-compatible for
    # existing experiment readers until their stored manifests are retired.
    payload['initial_loop_search'] = dict(payload['stage0_retrieval'])
    temporary = REPAIR_SUMMARY.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(REPAIR_SUMMARY)


def write_result_contract(output_dir, *, active_constraint_count,
                          balm_completed, balm_adopted):
    """Publish the terminal artifact set without a newest-directory race.

    The GUI deliberately consumes this contract instead of reimplementing the
    production pipeline or guessing which timestamped run finished last.
    Publishing is the final operation and uses an atomic rename, so a crashed
    or cancelled repair cannot be mistaken for a complete result.
    """
    if RESULT_JSON is None:
        return
    output_dir = Path(output_dir).resolve()
    artifacts = {
        'optimized_tum': output_dir / 'optimized_poses_tum.txt',
        'pose_graph_g2o': output_dir / 'pose_graph.g2o',
        'constraints_csv': output_dir / 'manual_loop_constraints.csv',
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            'cannot publish repair result; missing terminal artifact(s): '
            + ', '.join(missing)
        )
    payload = {
        'schema': 'lidar-map-refiner/headless-result',
        'schema_version': 1,
        'status': 'complete',
        'run_id': RUN_ID,
        'session': str(SESSION.resolve()),
        'mode': MODE,
        'environment': ENV,
        'terminal_output_dir': str(output_dir),
        **{name: str(path.resolve()) for name, path in artifacts.items()},
        'repair_summary_json': str(REPAIR_SUMMARY.resolve()),
        'proposal_ledger_jsonl': str(LEDGER_PATH.resolve()),
        'proposal_ledger_sha256': hashlib.sha256(
            LEDGER_PATH.read_bytes()
        ).hexdigest(),
        'active_constraint_count': int(active_constraint_count),
        'balm_completed': bool(balm_completed),
        'balm_adopted': balm_adopted,
        'input_sha256': dict(INPUT_SHA256),
        'output_sha256': {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in artifacts.items()
        },
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_JSON.with_name(RESULT_JSON.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(RESULT_JSON)


write_repair_summary('running')

def odometry_budget_measurement(result):
    s = result.preview.source_id
    t = result.preview.target_id
    odo_rel = np.linalg.inv(_ODO_T0[t]) @ _ODO_T0[s]
    corr = np.linalg.norm(
        (np.linalg.inv(odo_rel) @ result.transform_target_source_final)[:3, 3])
    chain = abs(_ODO_CUM[s] - _ODO_CUM[t])
    budget = BUDGET_BASE + BUDGET_RATE * chain
    return {
        'correction_m': float(corr),
        'chain_length_m': float(chain),
        'budget_m': float(budget),
        'budget_occupancy': float(corr / max(budget, 1e-12)),
    }


def odometry_budget_ok(result):
    measurement = odometry_budget_measurement(result)
    s = result.preview.source_id
    t = result.preview.target_id
    corr = measurement['correction_m']
    chain = measurement['chain_length_m']
    budget = measurement['budget_m']
    if corr > budget:
        log(f'[budget] {t}->{s}: correction {corr:.2f} m exceeds odometry '
            f'budget {budget:.2f} m (chain {chain:.0f} m) -> REJECT')
        return False
    return True


def gravity_consistency_measurement(result):
    if gravity is None:
        effect = gravity_consistency(
            result.transform_target_source_final, None, None,
            max_error_deg=GRAVITY_MAX_ERROR_DEG,
        )
    else:
        source_id = int(result.preview.source_id)
        target_id = int(result.preview.target_id)
        effect = gravity_consistency(
            result.transform_target_source_final,
            gravity[source_id], gravity[target_id],
            max_error_deg=GRAVITY_MAX_ERROR_DEG,
        )
    return {
        'available': bool(effect.available),
        'error_deg': effect.error_deg,
        'max_error_deg': float(effect.max_error_deg),
        'passed': bool(effect.passed),
        'reason': effect.reason,
    }


def gravity_consistency_ok(result):
    measurement = gravity_consistency_measurement(result)
    if not measurement['passed']:
        source_id = int(result.preview.source_id)
        target_id = int(result.preview.target_id)
        _LAST_GRAVITY_REASON[(source_id, target_id)] = measurement['reason']
        log(f'[gravity] {target_id}->{source_id}: '
            f'{measurement["reason"]} -> REJECT')
        return False
    return True


def append_proposal_audit(
    *, source, round_id, source_id, target_id, decision, reason,
    region_id=None, result=None, **extra,
):
    pid = proposal_id(
        RUN_ID, source, round_id, int(source_id), int(target_id), region_id
    )
    record = {
        'proposal_id': pid,
        'source': source,
        'round': int(round_id),
        'source_id': int(source_id),
        'target_id': int(target_id),
        'region_id': region_id,
        'decision': decision,
        'reason': reason,
        **extra,
    }
    if result is not None:
        transform = np.asarray(result.transform_target_source_final, dtype=float)
        record['gicp'] = {
            'fitness': float(result.fitness),
            'inlier_rmse_m': float(result.inlier_rmse),
            'transform_target_source': {
                'translation_m': [float(value) for value in transform[:3, 3]],
                'quaternion_xyzw': [
                    float(value) for value in matrix_to_quat_xyzw(transform)
                ],
            },
        }
        record['odometry_budget'] = odometry_budget_measurement(result)
        record['gravity_consistency'] = gravity_consistency_measurement(result)
        factor = _FACTOR_RESULT_CACHE.get((int(source_id), int(target_id)))
        if factor is not None:
            factor_transform = np.asarray(
                factor.transform_target_source, dtype=float
            )
            record['loop_factor'] = {
                'mode': LOOP_FACTOR_MODE,
                'valid': bool(factor.valid),
                'reason': factor.reason,
                'oriented_overlap': float(factor.overlap),
                'spatial_overlap': float(factor.spatial_overlap),
                'rmse_m': float(factor.inlier_rmse_m),
                'residual_scale_m': float(factor.residual_scale_m),
                'oriented_correspondences': int(
                    factor.orientation_consistent_count
                ),
                'observable_rank': int(factor.observable_rank),
                'condition_number': float(factor.scaled_condition_number),
                'transform_target_source': {
                    'translation_m': [
                        float(value) for value in factor_transform[:3, 3]
                    ],
                    'quaternion_xyzw': [
                        float(value)
                        for value in matrix_to_quat_xyzw(factor_transform)
                    ],
                },
                'information_upper_g2o': _information_upper(
                    factor.information_g2o
                ),
            }
    LEDGER.append(record)
    # A proposal audit contains all persistent scalar evidence needed by the
    # paper ledger.  The raw registration result still owns two point clouds;
    # retaining one for every accepted initial-loop proposal made the headless
    # parent grow by several GiB on NTU.  Release it after the record is
    # materialized.  The local caller can continue using its ``result`` object
    # for the current trial-PGO decision.
    gate_cache = globals().get('_LAST_GATE_RESULT')
    if isinstance(gate_cache, dict):
        gate_cache.pop((int(source_id), int(target_id)), None)
    return pid



# 被拒的假设里最接近的一个，按 (rmse, fitness) 记；gicp_gate 失败时报出来。
# 'SKIP (gate, full cascade)' 本身不说明任何事情：差 1 厘米和差 1 米都是这一句，
# 而这两种情况该做的事完全不同。诊断一次门限问题要靠单独写探针复现，
# 而转录出来的探针错了三次——让流水线自己说，比事后重建便宜也可靠。
_NEAR_MISS = {}
_LAST_GATE_RESULT = {}
_LAST_GRAVITY_REASON = {}


def _attempt(source_id, target_id, delta, cfg=None):
    try:
        r = workspace.run_gicp(source_id=source_id, target_id=target_id,
                               delta_transform_local=delta, config=cfg or reg_config)
    except Exception:
        return None
    key = (source_id, target_id)
    previous_result = _LAST_GATE_RESULT.get(key)
    if previous_result is None or r.inlier_rmse < previous_result.inlier_rmse:
        _LAST_GATE_RESULT[key] = r
    if r.fitness >= MIN_FITNESS and r.inlier_rmse <= MAX_RMSE:
        return r
    best = _NEAR_MISS.get(key)
    if best is None or r.inlier_rmse < best[0]:
        _NEAR_MISS[key] = (r.inlier_rmse, r.fitness)
    return None

def c2f_gate(source_id, target_id, deltas, tag):
    """由粗到细：每个初值走一条体素递减的链，只在最细那一档判决。

    **仅无头、benchmark 专用。** 由 `GHOSTLOOP_C2F` 触发（默认空 = 关闭），
    生产路径不走这里，GUI 也没有对应实现。它是第三份级联实现，只要存在，
    「GUI 与无头一致」这句话就有一个例外——所以对齐测试里把它显式登记为
    仅无头，而不是让检查对它沉默。

    对应半径跟着体素走（5 倍），否则粗体素配上细半径等于白粗——点稀了却还是只
    看很近的邻居。最后一档用预设体素，保证判决口径和验收门一致。
    """
    STATS['gicp_pairs'] += 1
    # 链条就是给定的那一串，末档即工作分辨率。此前这里追加了一次预设体素，
    # 于是室外变成 0.5 -> 0.4 -> 0.3 -> 0.4：最后一步又退回粗档，把前一步细化
    # 出来的结果重新糊掉，且白花一次配准。只在预设比末档更细时才追加。
    schedule = list(C2F)
    if not schedule or repair_preset['voxel'] < schedule[-1]:
        schedule.append(repair_preset['voxel'])
    best, where = None, ''
    for k, seed in enumerate(deltas):
        cur, r = seed, None
        for step, v in enumerate(schedule):
            cfg = replace(reg_config, voxel_size=v,
                          max_correspondence_distance=max(2.0, 5.0 * v),
                          max_iterations=60 if step < len(schedule) - 1 else 30)
            try:
                r = workspace.run_gicp(source_id=source_id, target_id=target_id,
                                       delta_transform_local=cur, config=cfg)
            except Exception:
                r = None
                break
            cur = (np.linalg.inv(workspace.trajectory.transforms_world_sensor[source_id])
                   @ workspace.trajectory.transforms_world_sensor[target_id]
                   @ r.transform_target_source_final)
        if r is None:
            continue
        if r.fitness >= MIN_FITNESS and r.inlier_rmse <= MAX_RMSE:
            if best is None or r.inlier_rmse < best.inlier_rmse:
                best, where = r, f' (c2f, seed {k})'
        else:
            key = (source_id, target_id)
            prev = _NEAR_MISS.get(key)
            if prev is None or r.inlier_rmse < prev[0]:
                _NEAR_MISS[key] = (r.inlier_rmse, r.fitness)
    if best is None:
        miss = _NEAR_MISS.pop((source_id, target_id), None)
        detail = (f'best rmse={miss[0]:.3f} (fit={miss[1]:.3f}) vs bound {MAX_RMSE:.3f}'
                  if miss else 'no hypothesis produced a result')
        log(f'[{tag}] {target_id}->{source_id}: SKIP (gate, c2f) -- {detail}')
        return None
    _NEAR_MISS.pop((source_id, target_id), None)
    if not odometry_budget_ok(best):
        return None
    if not gravity_consistency_ok(best):
        return None
    log(f'[{tag}] {target_id}->{source_id}: fit={best.fitness:.3f} '
        f'rmse={best.inlier_rmse:.3f} -> ACCEPT{where}')
    return best


def gicp_gate(source_id, target_id, deltas, tag, seed_stage=False):
    _LAST_GATE_RESULT.pop((source_id, target_id), None)
    if C2F:
        return c2f_gate(source_id, target_id, deltas, tag)
    STATS['gicp_pairs'] += 1
    if not isinstance(deltas, list):
        deltas = [deltas]
    # Four tiers, tried in increasing cost; within a tier every hypothesis is
    # scored and the smallest residual wins. The first two tiers differ only in
    # correspondence radius, and they are not two tries at the same thing: the
    # collocation seed zeroes the translation error, so what is left is the
    # descriptor's yaw quantisation (one sector, 6 deg), and that turns into
    # point displacement in proportion to range -- 1.0 m at 10 m, 3.1 m at 30 m.
    # A near radius resolves the close case cleanly and cannot reach the far
    # one; a wide radius reaches the far one but pairs points across faces when
    # the seed was already good. GT-checked on MCD kth_night_01: 970->571 and
    # 841->689 pass at 2 m (rmse 0.338 / 0.250) and FAIL at 8 m (1.912 / 0.664),
    # while 1309->358 fails at 2 m (0.564) and passes at 8 m (0.228).
    #
    # The remaining two tiers are not more of the same. `submap` accumulates ten
    # frames on the SOURCE side: on sparse scans a single frame simply does not
    # carry enough points to reach overlap 0.5, whatever the radius, and every
    # constraint the 16-ring Gazebo sequence recovers comes from this tier.
    #
    # `anneal` is a TWO-STAGE tier: register once with the wide configuration to
    # get a warm start, then re-register from that result with the near one and
    # judge on the second. It reaches seeds the collocation prior cannot -- an
    # odometry seed under heavy drift, where a single wide pass lands close
    # enough for the near radius to finish the job.
    #
    # 2026-08-06: this tier used to differ from `wide` by more than its schedule
    # -- the warm-up ran with the orientation gate off, so it could move where
    # normals could not yet agree. That gate is gone from this version (held for
    # the journal version, see the 2026-08-06 snapshot), so `anneal_config` and
    # `wide_config` are now identical and only the two-stage schedule remains.
    # It is kept because it earned 27 initial-loop and 29 round constraints across
    # the logged sessions, but whether the schedule alone still earns them is an
    # open question this version's experiments must answer.
    #
    # Across the session logs main wins ~1285 times, submap 40, anneal 13,
    # wide 9 -- all measured WITH the gate; treat them as historical.
    #
    # What did go is the coarse-ICP prescreen. Ordering only mattered because
    # the cascade stopped at its first success, so a cheap registration had to
    # guess which hypothesis to try first; taking the smallest residual within a
    # tier decides the same question better, and for 13.6 s per candidate less.
    # A tier's hypotheses are independent -- same pair, different starting
    # transform -- and their KD queries are single-threaded (see concurrency.py),
    # so concurrent registration improves throughput. Native preprocessing may
    # still use internal threads, which is why the shared policy caps fan-out.
    # Selection stays on the smallest residual with the seed index breaking
    # ties, which makes the result identical to the serial order.
    def _pick(results, name):
        out, which = None, ''
        for k, c in enumerate(results):
            if c is not None and (out is None or c.inlier_rmse < out.inlier_rmse):
                out, which = c, f' ({name}, seed {k})'
        return out, which

    def best_of(cfg, name):
        if len(deltas) < 2 or HYPOTHESIS_WORKERS < 2:
            return _pick([_attempt(source_id, target_id, d, cfg) for d in deltas], name)
        with ThreadPoolExecutor(max_workers=min(HYPOTHESIS_WORKERS, len(deltas))) as ex:
            futures = [ex.submit(_attempt, source_id, target_id, d, cfg) for d in deltas]
            return _pick([f.result() for f in futures], name)

    # Initial Loop Search skips `submap`. Across every real sequence's logs it has won 277
    # near / 27 annealed / 22 wide / **0 submap** seed constraints, while
    # costing 5.1 s of the 8.3 s a failing candidate spends in the cascade
    # (61%) -- the source side accumulates 21 frames, so it carries twenty
    # times the points. Only the Gazebo simulation ever won a seed there (12),
    # and its scans are 16-ring: exactly the sparsity this tier exists for.
    # The ghost rounds keep it; they win 12 real constraints with it (MCD ntu
    # 795->1751 and 1609->1867), and they propose far fewer pairs, so the same
    # per-failure cost buys much less waste.
    tiers = [(reg_config, 'near'), (wide_config, 'wide')]
    if not seed_stage:
        tiers.append((submap_config, 'submap'))
    r, sweep = None, ''
    for cfg, name in tiers:
        r, sweep = best_of(cfg, name)
        if r is not None:
            break
    if r is None:
        def _annealed(delta):
            try:
                warm = workspace.run_gicp(source_id=source_id, target_id=target_id,
                                          delta_transform_local=delta,
                                          config=anneal_config)
            except Exception:
                return None
            seed = (np.linalg.inv(workspace.trajectory.transforms_world_sensor[source_id])
                    @ workspace.trajectory.transforms_world_sensor[target_id]
                    @ warm.transform_target_source_final)
            return _attempt(source_id, target_id, seed)

        if len(deltas) < 2 or HYPOTHESIS_WORKERS < 2:
            warmed = [_annealed(d) for d in deltas]
        else:
            with ThreadPoolExecutor(
                    max_workers=min(HYPOTHESIS_WORKERS, len(deltas))) as ex:
                warmed = [f.result() for f in [ex.submit(_annealed, d) for d in deltas]]
        r, sweep = _pick(warmed, 'annealed')
    if r is None:
        miss = _NEAR_MISS.pop((source_id, target_id), None)
        detail = (f'best rmse={miss[0]:.3f} (fit={miss[1]:.3f}) vs bound {MAX_RMSE:.3f}'
                  if miss else 'no hypothesis produced a result')
        log(f'[{tag}] {target_id}->{source_id}: SKIP (gate, full cascade) -- {detail}')
        return None
    _NEAR_MISS.pop((source_id, target_id), None)
    if not odometry_budget_ok(r):
        return None
    if not gravity_consistency_ok(r):
        return None
    log(f'[{tag}] {target_id}->{source_id}: fit={r.fitness:.3f} rmse={r.inlier_rmse:.3f} '
        f'-> ACCEPT{sweep}')
    if LOOP_FACTOR_MODE == 'oriented_anisotropic':
        factor = oriented_factor_for_registration(r)
        _FACTOR_RESULT_CACHE[(int(source_id), int(target_id))] = factor
        if not factor.valid:
            _LAST_FACTOR_REASON[(int(source_id), int(target_id))] = factor.reason
            log(f'[{tag}] {target_id}->{source_id}: oriented factor '
                f'{factor.reason} -> REJECT')
            return None
        log(f'[{tag}] {target_id}->{source_id}: oriented factor '
            f'overlap={factor.overlap:.3f} rmse={factor.inlier_rmse_m:.3f} m '
            f'rank={factor.observable_rank} -> PROBATION')
    return r


def oriented_factor_for_registration(result):
    source_id = int(result.preview.source_id)
    target_id = int(result.preview.target_id)
    for frame_id in (source_id, target_id):
        if frame_id not in _ORIENTED_CLOUD_CACHE:
            _ORIENTED_CLOUD_CACHE[frame_id] = prepare_oriented_cloud(
                workspace.load_local_points(frame_id), ORIENTED_FACTOR_CONFIG
            )
    source = _ORIENTED_CLOUD_CACHE[source_id]
    target = _ORIENTED_CLOUD_CACHE[target_id]
    return estimate_oriented_surface_factor(
        source.points,
        target.points,
        result.transform_target_source_final,
        replace(ORIENTED_FACTOR_CONFIG, voxel_size_m=0.0),
        source_normals_local=source.normals,
        target_normals_local=target.normals,
    )


def constraint_row(result, enabled=True):
    factor = _FACTOR_RESULT_CACHE.get((
        int(result.preview.source_id), int(result.preview.target_id)
    ))
    tr = (
        factor.transform_target_source
        if LOOP_FACTOR_MODE == 'oriented_anisotropic' and factor is not None
        else result.transform_target_source_final
    )
    q = matrix_to_quat_xyzw(tr)
    row = [('1' if enabled else '0'),
           str(result.preview.source_id), str(result.preview.target_id),
           *(f'{v:.12f}' for v in tr[:3, 3]), *(f'{v:.12f}' for v in q),
           SIGMA_T, SIGMA_T, SIGMA_T, SIGMA_R, SIGMA_R, SIGMA_R]
    if LOOP_FACTOR_MODE == 'oriented_anisotropic':
        if factor is None or not factor.valid:
            raise RuntimeError('oriented factor missing for accepted registration')
        row.append(json.dumps(_information_upper(factor.information_g2o)))
    return row

_PREP_CLOUDS = None

def diagnose(traj, max_regions=None):
    # local-frame clouds are pose-independent: prepare once, reuse every round
    global _PREP_CLOUDS
    if _PREP_CLOUDS is None:
        _PREP_CLOUDS = [
            prepare_local_cloud(workspace.load_local_points(i), balm_params)
            for i in range(traj.size)
        ]
    kw = {} if max_regions is None else {'max_regions': int(max_regions)}
    return detect_ghost_regions(_PREP_CLOUDS, traj.transforms_world_sensor,
                                balm_params,
                                max_observation_range=GHOST_RANGE, **kw)


def solve_constraint_rows(rows_to_solve, output_dir, *, csv_name='manual_loop_constraints.csv'):
    """Solve from the immutable original graph and original trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    full_information = any(
        len(row) > len(CONSTRAINT_HEADER) and row[len(CONSTRAINT_HEADER)]
        for row in rows_to_solve
    )
    header = list(CONSTRAINT_HEADER)
    if full_information:
        header.append('information_upper_json')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows_to_solve:
            normalized = list(row)
            if full_information and len(normalized) == len(CONSTRAINT_HEADER):
                normalized.append('')
            writer.writerow(normalized)
    g2o_in = output_dir / 'edited_input_pose_graph.g2o'
    g2o_in.write_bytes((SESSION / 'pose_graph.g2o').read_bytes())
    completed = subprocess.run(
        [PY, '-u', str(CLI_DIR / 'cli.py'),
         '--session-root', str(SESSION), '--g2o', str(g2o_in),
         '--tum', str(SESSION / 'optimized_poses_tum.txt'),
         '--keyframe-dir', str(KF_DIR), '--constraints-csv', str(csv_path),
         '--output-dir', str(output_dir),
         '--optimize-mode', PGO_OPTIMIZE_MODE,
         '--manual-information-scale', str(PGO_INFORMATION_SCALE),
         '--loop-correlation-window-keyframes', str(PGO_CORRELATION_WINDOW),
         '--loop-cluster-information-budget', str(PGO_CLUSTER_BUDGET),
         '--loop-correlation-policy', PGO_CORRELATION_POLICY,
         '--loop-cluster-allocation', PGO_CLUSTER_ALLOCATION,
         '--skip-map-build'],
        capture_output=True, text=True,
    )
    return completed


def trial_graph_audit(rows_before, candidate_row, trial_dir, base_tum_path):
    """Solve one probationary factor from the immutable odometry graph.

    ``base_tum_path`` represents ``rows_before``. Carrying the most recent
    retained trial forward avoids an unnecessary second solve while keeping
    every trial rooted in the unmodified input graph and trajectory.
    """
    trial_rows = [list(row) for row in rows_before] + [list(candidate_row)]
    completed = solve_constraint_rows(trial_rows, trial_dir)
    if completed.returncode != 0:
        return {
            'completed': completed,
            'rows': trial_rows,
            'graph_ok': False,
            'candidate_residual_m': None,
            'worst_factor_residual_m': None,
            'base_worst_factor_residual_m': None,
            'candidate_factor_nis': None,
            'candidate_consistency_score': None,
            'worst_factor_consistency_score': None,
            'base_worst_factor_consistency_score': None,
        }

    trial_details = factor_residual_details(
        trial_dir / 'optimized_poses_tum.txt', trial_rows)
    candidate_detail = trial_details[-1]
    candidate_residual = float(candidate_detail['translation_m'])
    candidate_score = float(candidate_detail['gate_value'])
    trial_worst_score = float(max(
        item['gate_value'] for item in trial_details
    ))
    trial_worst_residual = float(max(
        item['translation_m'] for item in trial_details
    ))
    base_details = factor_residual_details(base_tum_path, rows_before)
    base_worst_score = float(max(
        (item['gate_value'] for item in base_details), default=0.0
    ))
    base_worst_residual = float(max(
        (item['translation_m'] for item in base_details), default=0.0
    ))
    effect = evaluate_graph_trial(
        candidate_score, trial_worst_score, base_worst_score,
        residual_limit_m=CONSISTENCY_LIMIT,
    )
    return {
        'completed': completed,
        'rows': trial_rows,
        'graph_ok': effect.retained,
        'graph_reason': effect.reason,
        'candidate_residual_m': candidate_residual,
        'worst_factor_residual_m': trial_worst_residual,
        'base_worst_factor_residual_m': base_worst_residual,
        'candidate_factor_nis': float(candidate_detail['nis']),
        'candidate_consistency_score': candidate_score,
        'worst_factor_consistency_score': trial_worst_score,
        'base_worst_factor_consistency_score': base_worst_score,
    }

# ---------- Initial Loop Search: descriptor retrieval and geometric audit ----------
log(f'== MODE: {MODE} ==')
log('== Initial Loop Search: descriptor retrieval and geometric audit ==')
gravity = None
if config.gravity_canonicalization_enable:
    try:
        gravity = load_scan_context_gravity(SESSION, trajectory)
    except Exception as exc:
        log(f'[InitialLoops] no gravity sidecar ({exc}); matching without canonicalization')
if MODE == 'production':
    descriptors, masks = build_descriptor_stack(
        workspace.load_local_points_uncached,
        trajectory.size,
        config,
        gravity,
        log_fn=log,
    )
    _PREP_CLOUDS = None
else:
    descriptors, masks, _PREP_CLOUDS = build_descriptor_and_prepared_stack(
        workspace.load_local_points_uncached,
        lambda points: prepare_local_cloud(points, diagnosis_params),
        trajectory.size,
        config,
        gravity,
        log_fn=log,
    )

def colloc_desired_rotation(sid, tid, yaw_deg):
    """Gravity-canonical desired target->source rotation for seed inits: align
    both frames' measured up-vectors, then apply the SC yaw about gravity. A
    plain Rz(yaw) inherits the source's drifted attitude — at MCD ntu's tail
    (tens of degrees of accumulated pitch riding on the z-drift) that alone
    kills the end-to-start loop, which registers at rmse 0.23 once
    gravity-corrected."""
    yaw = math.radians(yaw_deg)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                   [math.sin(yaw), math.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    if gravity is None:
        return Rz
    Rg_s = _gravity_canonical_rotation(gravity[sid])
    Rg_t = _gravity_canonical_rotation(gravity[tid])
    return Rg_t.T @ Rz @ Rg_s
log(f'[InitialLoops] env={ENV} dual_z={config.dual_z_layer_enable} '
    f'weights={CHANNEL_WEIGHTS} radius={config.max_radius:.0f} '
    f'min_joint_rings={config.min_joint_rings} '
    f'sector_support_exp={config.sector_support_exponent:.2f}')
if ENV == 'indoor' and not config.dual_z_layer_enable:
    log('[InitialLoops] indoor session recorded dual_z=false; using protected '
        'single-layer retrieval instead of silently claiming UpDownSC')
# Recall-first: a loose sanity cap on descriptor distance instead of a hard
# cutoff (GT-verified TRUE loops score d=0.55-0.62 on vegetation-heavy campus
# scenes, e.g. MCD ntu 70->649 at d=0.60). The gate cascade, odometry budget,
# probation and repair check carry precision; retrieval only has to surface
# candidates. Both numbers live in repair_presets so the GUI cannot disagree.
_SEED_N = repair_preset['seed_max_candidates']
_SEED_D = repair_preset['seed_max_distance']
_candidate_pool_override = int(os.environ.get('GHOSTLOOP_CANDIDATE_POOL', '0'))
if _candidate_pool_override > 0:
    _SEED_N = _candidate_pool_override
_candidate_distance_override = os.environ.get('GHOSTLOOP_CANDIDATE_DISTANCE')
if _candidate_distance_override:
    _SEED_D = float(_candidate_distance_override)
_SEED_SEGMENT_RADIUS = int(os.environ.get(
    'GHOSTLOOP_CANDIDATE_SEGMENT_RADIUS',
    min(100, max(10, trajectory.size // 8)),
))
_SEED_RING_KEY_TOP_K = int(os.environ.get(
    'GHOSTLOOP_RING_KEY_TOP_K', '10'
))


def _seed_deltas(sid, tid):
    """Initial Loop Search guesses: collocation per yaw peak, plus an
    odometry-position seed for the best peak."""
    peaks = pair_yaw_peaks(
        descriptors, masks, sid, tid, CHANNEL_WEIGHTS, config.num_rings,
        min_joint_rings=config.min_joint_rings,
        retrieval_height_offset=config.retrieval_height_offset,
        sector_support_exponent=config.sector_support_exponent)
    relative = np.linalg.inv(T[tid]) @ T[sid]
    out = []
    for _, yaw_deg in peaks:
        desired = colloc_desired_rotation(sid, tid, yaw_deg)
        if not out:
            d = np.eye(4)
            d[:3, :3] = relative[:3, :3].T @ desired
            out.append(d)
        d4 = np.eye(4)
        d4[:3, :3] = desired
        out.append(np.linalg.inv(T[sid]) @ T[tid] @ d4)
    return out


if INITIAL_CONSTRAINTS_CSV is None:
    seeds = find_seed_candidates(
        descriptors, masks, max_seeds=_SEED_N, max_distance=_SEED_D,
        min_index_gap=min(150, trajectory.size // 4),
        segment_radius=_SEED_SEGMENT_RADIUS,
        ring_key_top_k=_SEED_RING_KEY_TOP_K,
        channel_weights=CHANNEL_WEIGHTS, num_rings=config.num_rings,
        min_joint_rings=config.min_joint_rings,
        retrieval_height_offset=config.retrieval_height_offset,
        sector_support_exponent=config.sector_support_exponent,
        log_fn=log)
    log(f'[InitialLoops] {len(seeds)} candidates: '
        + ', '.join(f'{s.target_id}->{s.source_id}(d={s.distance:.2f},yaw={s.yaw_deg:.0f})'
                    for s in seeds))
else:
    seeds = []
    log(f'[InitialLoops] fixed checkpoint: {INITIAL_CONSTRAINTS_CSV} '
        '(retrieval and initial-loop GICP skipped)')

if MODE == 'production':
    baseline_regions = []
    baseline = 0.0
    log('[InitialLoops] production path: duplicated-surface audit is '
        'diagnostic-only and is skipped during loop admission')
else:
    baseline_regions = diagnose(trajectory)
    baseline = ghost_badness(baseline_regions)
    log(f'[InitialLoops] pre-search baseline: badness={baseline:.1f} '
        f'({len(baseline_regions)} regions)')

rows = (
    load_initial_constraints_csv(INITIAL_CONSTRAINTS_CSV, trajectory.size)
    if INITIAL_CONSTRAINTS_CSV is not None else []
)
row_proposal_ids = []
for checkpoint_index, row in enumerate(rows):
    row_proposal_ids.append(append_proposal_audit(
        source='stage0_checkpoint', round_id=0,
        source_id=int(row[1]), target_id=int(row[2]),
        decision='loaded_probationary',
        reason='loaded from frozen initial-loop checkpoint',
        checkpoint_row=checkpoint_index,
    ))
seed_row_index = list(range(len(rows)))
INITIAL_CONSTRAINT_COUNT = len(rows)
write_repair_summary('running')
T = trajectory.transforms_world_sensor

# The following support gate belongs only to explicit legacy modes. Round
# proposals are generated FROM ghost regions, so they cannot fire where
# the map is already consistent. Initial-loop candidates have no such discipline:
# retrieval proposes wherever two places look alike, defect or not. On Oxford
# Spires christ-church-05 that is how a constraint entered a stretch the
# diagnosis never flagged -- it corrected the graph by 25 cm where the odometry
# was right to 3.5 cm, and the session got worse.
#
# The seeds still have to be allowed to do their real job, which is the loops
# the detector physically CANNOT see: two layers stop registering as a ghost
# past GHOST_MAX_SEP, so a 74 m revisit (MCD kth_night_01) leaves no ghost to
# find and would be destroyed by a naive "must be diagnosed" rule. Hence the
# split: a large correction is self-justifying, a small one must be one the map
# actually asked for.
# Ghost layers stop registering as such past this separation, so a correction
# larger than it describes an error the diagnosis could not have seen and the
# seed is self-justifying (MCD kth_night_01 owes 64% of its recovery to a
# single 74 m revisit, which leaves no ghost at all). Below it the diagnosis
# had every opportunity to speak; if it did not, there is nothing there to fix.
# Production bypasses this map-support gate entirely and performs BALM once,
# after the audited PGO. ``ghost`` retains the gate and proposal loop only for
# exact historical comparison.

GHOST_MAX_SEP = REPAIR_POLICY['ghost_max_separation_m']
SEED_SUPPORT_TOL = REPAIR_POLICY['seed_support_tolerance_kf']

# correction_exceeds_own_noise 已于 2026-08-06 删除（此前标着 NOT CALLED）。
# 它想用「修正量必须大于自身 inlier_rmse」判约束该不该收，但 2026-08-05 用真值
# 标注 54 条约束测出 rmse 与真实误差相关系数只有 +0.144 —— 拿一个与正确性无关的
# 量当分母，判出来的是噪声。完整证据见
# icra2027_runtime/experiments/nightly_20260805/ANALYSIS.md 第二节。


def seed_is_warranted(sid, tid, result):
    """Would the map have asked for this seed?

    Round proposals are generated FROM ghost regions and so cannot fire on a
    consistent map. Initial-loop candidates had no such discipline: retrieval proposes
    wherever two places look alike. On Oxford Spires christ-church-05 that let
    a constraint into a stretch the diagnosis never flagged -- it corrected the
    graph by 25 cm where the odometry was right to 3.5 cm, and both the
    trajectory (16.6 -> 25.6 cm) and the map against an independent survey scan
    (9.9 -> 12.5 cm) got worse. With this check the same session ends at
    16.8 cm, i.e. essentially untouched.
    """
    odo = np.linalg.inv(T[tid]) @ T[sid]
    corr = np.linalg.norm(
        (np.linalg.inv(odo) @ result.transform_target_source_final)[:3, 3])
    if corr > GHOST_MAX_SEP:
        return True
    for reg in baseline_regions:
        pair = reg.get('suggested_pair')
        if not pair:
            continue
        a, b = int(pair[0]), int(pair[1])
        if ((abs(a - sid) <= SEED_SUPPORT_TOL and abs(b - tid) <= SEED_SUPPORT_TOL)
                or (abs(a - tid) <= SEED_SUPPORT_TOL and abs(b - sid) <= SEED_SUPPORT_TOL)):
            return True
    log(f'[seed-need] {sid}->{tid}: corrects {corr*100:.1f} cm, inside the '
        f'detector\'s {GHOST_MAX_SEP*100:.0f} cm reach, yet no ghost was '
        f'diagnosed there -> nothing to fix, REJECT')
    return False
if MODE in ('ba-only', 'radius') and INITIAL_CONSTRAINTS_CSV is None:
    seeds = []
if SEED_WIDEST and len(seeds) > 1:
    ranked = sorted(
        seeds, key=lambda item: abs(item.source_id - item.target_id), reverse=True
    )
    selected = ranked[min(SEED_WIDEST, len(ranked)) - 1]
    log(f'[InitialLoops] widest-only control selected '
        f'{selected.target_id}->{selected.source_id} '
        f'(span {abs(selected.source_id - selected.target_id)})')
    seeds = [selected]

STATS['proposed'] += len(seeds)
stage0_trial_root = SESSION / 'manual_loop_runs' / (
    datetime.now().strftime('%Y%m%d_%H%M%S') + '_stage0_trials'
)
stage0_base_tum = SESSION / 'optimized_poses_tum.txt'
for s in seeds:
    _cap = SEED_CAP or PER_ROUND_CAP
    if _cap and len(seed_row_index) >= _cap:
        log(f'[InitialLoops] candidate cap {_cap} reached, deferring the rest')
        break
    r = gicp_gate(s.source_id, s.target_id,
                  _seed_deltas(s.source_id, s.target_id), 'InitialLoops',
                  seed_stage=True)
    if r is None:
        attempted = _LAST_GATE_RESULT.get((s.source_id, s.target_id))
        factor_reason = _LAST_FACTOR_REASON.pop(
            (s.source_id, s.target_id), None
        )
        gravity_reason = _LAST_GRAVITY_REASON.pop(
            (s.source_id, s.target_id), None
        )
        append_proposal_audit(
            source='stage0_descriptor', round_id=0,
            source_id=s.source_id, target_id=s.target_id,
            decision='rejected', reason=(
                f'oriented factor gate failed: {factor_reason}'
                if factor_reason else (
                    f'gravity consistency gate failed: {gravity_reason}'
                    if gravity_reason else
                    'GICP or odometry-budget gate failed'
                )
            ),
            result=attempted, gicp_gate_passed=False,
            descriptor_distance=float(s.distance),
            descriptor_yaw_deg=float(s.yaw_deg),
        )
        continue
    if MODE != 'production' and not seed_is_warranted(
            s.source_id, s.target_id, r):
        append_proposal_audit(
            source='stage0_descriptor', round_id=0,
            source_id=s.source_id, target_id=s.target_id,
            decision='rejected',
            reason='small correction lacked map-inconsistency support',
            result=r, gicp_gate_passed=True,
            descriptor_distance=float(s.distance),
            descriptor_yaw_deg=float(s.yaw_deg),
        )
        continue
    candidate_row = constraint_row(r)
    pid = proposal_id(
        RUN_ID, 'stage0_descriptor', 0, s.source_id, s.target_id
    )
    trial_dir = stage0_trial_root / pid
    trial = trial_graph_audit(rows, candidate_row, trial_dir, stage0_base_tum)
    if trial['completed'].returncode != 0:
        append_proposal_audit(
            source='stage0_descriptor', round_id=0,
            source_id=s.source_id, target_id=s.target_id,
            decision='rejected', reason='trial PGO failed',
            result=r, gicp_gate_passed=True,
            descriptor_distance=float(s.distance),
            descriptor_yaw_deg=float(s.yaw_deg),
        )
        continue
    if not trial['graph_ok']:
        append_proposal_audit(
            source='stage0_descriptor', round_id=0,
            source_id=s.source_id, target_id=s.target_id,
            decision='retracted',
            reason='trial factor triggered graph residual audit',
            result=r, gicp_gate_passed=True,
            descriptor_distance=float(s.distance),
            descriptor_yaw_deg=float(s.yaw_deg),
            factor_residual_m=trial['candidate_residual_m'],
            worst_factor_residual_m=trial['worst_factor_residual_m'],
            base_worst_factor_residual_m=(
                trial['base_worst_factor_residual_m']),
            factor_nis=trial['candidate_factor_nis'],
            factor_consistency_score=trial['candidate_consistency_score'],
            worst_factor_consistency_score=(
                trial['worst_factor_consistency_score']),
            base_worst_factor_consistency_score=(
                trial['base_worst_factor_consistency_score']),
        )
        continue

    seed_row_index.append(len(rows))
    rows.append(candidate_row)
    row_proposal_ids.append(append_proposal_audit(
        source='stage0_descriptor', round_id=0,
        source_id=s.source_id, target_id=s.target_id,
        decision='admitted_probationary',
        reason='passed trial PGO; awaiting post-solve audit',
        result=r, gicp_gate_passed=True,
        descriptor_distance=float(s.distance),
        descriptor_yaw_deg=float(s.yaw_deg),
        factor_residual_m=trial['candidate_residual_m'],
        worst_factor_residual_m=trial['worst_factor_residual_m'],
        base_worst_factor_residual_m=trial['base_worst_factor_residual_m'],
        factor_nis=trial['candidate_factor_nis'],
        factor_consistency_score=trial['candidate_consistency_score'],
        worst_factor_consistency_score=trial['worst_factor_consistency_score'],
        base_worst_factor_consistency_score=(
            trial['base_worst_factor_consistency_score']),
    ))
    stage0_base_tum = trial_dir / 'optimized_poses_tum.txt'
    STATS['accepted'] += 1
log(f'== Initial Loop Search done: {len(seed_row_index)} accepted constraints ==')
if not rows and MODE == 'ghost':
    baseline_now = ghost_badness(diagnose(trajectory))
    if baseline_now > 3000.0:
        # heavily drifted and retrieval-blind: diagnosis alone cannot bridge
        # large-drift loops (range-limited); a human seed is genuinely needed
        log('No seeds accepted on a heavily drifted session; aborting '
            '(manual seed needed).')
        write_repair_summary(
            'manual_seed_required',
            reason='no accepted seed on a heavily drifted session',
            baseline_ghost_badness=baseline_now,
            active_constraint_count=0,
        )
        raise SystemExit(0)
    log('No seeds accepted; map is near-consistent — proceeding with '
        'diagnosis-driven rounds only.')

# ---------- Repair rounds ----------

CSV_HEADER = CONSTRAINT_HEADER
seen_pairs = {(row[1], row[2]) for row in rows}
probation_done = False
# Which ghost each round constraint was supposed to close, so the next
# round can check whether it actually closed it.
current_traj = trajectory
prev_badness = baseline
stage0_audited_out = None
production_balm_completed = False
production_balm_adopted = None
for round_id in range(1, MAX_ROUNDS + 1):
    out = SESSION / 'manual_loop_runs' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + f'_auto_r{round_id}')
    log(f'== Round {round_id}: optimizing with {sum(r[0]=="1" for r in rows)} constraints ==')
    r1 = solve_constraint_rows(rows, out)
    if r1.returncode != 0:
        log(r1.stdout[-1500:]); log(r1.stderr[-1500:]); raise SystemExit('optimizer failed')

    # Post-optimization consistency, the one criterion here that carries no
    # dataset-dependent constant. Every gate before this point judges a pair on
    # its own; this one asks whether the accepted set agrees with itself. When
    # two constraints contradict, the optimizer splits the difference and BOTH
    # show a large residual, so the flagged set is resolved by leave-one-out:
    # the constraint whose removal most reduces the worst residual is the one
    # that was wrong.
    flagged = [] if NO_CONSISTENCY else [
        (v, i) for v, i in factor_residuals(out / 'optimized_poses_tum.txt', rows)
        if v > CONSISTENCY_LIMIT]
    # Any single constraint the solution cannot satisfy is actionable. An
    # earlier version required two, on the theory that a wrong loop always
    # drags a conflicting right one over the threshold with it; MCD
    # tuhh_night_08 disproved that -- all three of its constraints were wrong
    # by 2-4 m against ground truth, yet only one crossed the bound and the
    # check never ran.
    if flagged:
        metric_label = 'normalized NIS'
        log(f'[consistency] {len(flagged)} constraint(s) disagree after optimization '
            f'(worst {max(flagged)[0]:.2f} {metric_label}) -> leave-one-out')
        best = None
        for _, idx in sorted(flagged, reverse=True)[:4]:
            trial = [list(r) for r in rows]
            trial[idx][0] = '0'
            trial_dir = out.parent / (out.name + f'_loo{idx}')
            rc = solve_constraint_rows(trial, trial_dir, csv_name=f'loo{idx}.csv')
            if rc.returncode != 0:
                continue
            worst = max(v for v, _ in factor_residuals(
                trial_dir / 'optimized_poses_tum.txt', trial))
            if best is None or worst < best[0]:
                best = (worst, idx)
        if best is not None and best[0] < max(flagged)[0] * 0.5:
            rows[best[1]][0] = '0'
            append_proposal_audit(
                source='post_solve_consistency', round_id=round_id,
                source_id=int(rows[best[1]][1]), target_id=int(rows[best[1]][2]),
                decision='retracted',
                reason='leave-one-out halved the worst factor residual',
                original_proposal_id=row_proposal_ids[best[1]],
                factor_residual_before_m=None,
                factor_residual_after_m=None,
                factor_consistency_score_before=float(max(flagged)[0]),
                factor_consistency_score_after=float(best[0]),
            )
            log(f'[consistency] disabled constraint {rows[best[1]][2]}->'
                f'{rows[best[1]][1]}: worst residual {max(flagged)[0]:.2f} -> '
                f'{best[0]:.2f} normalized NIS without it')
            continue

    if MODE == 'production':
        stage0_audited_out = out
        if STAGE0_AUDIT_ONLY:
            log('== Initial-loop audit-only control complete; Final Map Refinement skipped ==')
            break
        r2, balm_out = run_balm_cli(out, '_balm_final')
        if r2.returncode != 0:
            raise SystemExit('final BALM failed')
        production_balm_adopted = adopt_balm_or_restore_pgo(out, balm_out)
        production_balm_completed = True
        out = balm_out
        log('== production complete: one final double-sided BALM pass; '
            'residual map inconsistencies are diagnostic-only ==')
        break

    if MODE == 'ba-only':
        r2, balm_out = run_balm_cli(out, '_balm_control')
        if r2.returncode != 0:
            raise SystemExit('BALM-only control failed')
        log('== ba-only control complete; BALM output is not fed back ==')
        break

    # Core ICRA path: diagnose the PGO map directly.  BALM is a separate
    # control and cannot feed evidence back into the proposal loop.
    balm_out = out
    solved_traj = load_tum_trajectory(out / 'optimized_poses_tum.txt')
    regions = diagnose(solved_traj)
    badness = ghost_badness(regions)
    prev_badness = badness
    log(f'== Round {round_id} result: {len(regions)} map-inconsistency '
        f'hypotheses, badness={badness:.1f} (PGO poses) ==')

    # 修复效果验证（rounds 之后检查"这条约束有没有闭上它瞄准的鬼影"）已于
    # 2026-08-05 删除。六次出手全部误伤，一次未中。
    #
    # 判据是拿地图一致性代理轨迹正确性。同一晚的数据直接反驳了这个代理关系
    # ——MCD ntu 逐轮：r4 ATE 232.8 cm / badness 6378，r6 ATE 272.4 cm / badness 483。
    # 地图越来越一致，轨迹越来越差。
    #
    # 它撤掉的约束用真值逐条判（真值间距 = 两帧在 GT 里的实际距离）：
    #     475->685  4.50 m   788->831  1.83 m   795->1751 2.61 m   392->769 4.46 m
    # 四条全是同一地点的真回环。另两次在 in-house 会话上（无真值），按检测器自己的
    # 1 米体素口径量，那条约束把两层从 30.0 cm 拉到 19.7 cm、少数层 105->258 点、
    # 支持帧 51->86 —— 正在起作用，却被判为无效。
    #
    # 前提也站不住：一条正确的回环没能闭上某个鬼影是正常的。鬼影可能由整条回路上
    # 分散的漂移造成，一条约束只修其中一部分；几个鬼影也可能同源。
    #
    # 代价确凿：每触发一次多跑一轮 PGO+BALM。论文里最好的那次运行（2026-07-29，
    # 16 条约束）根本没有它，今晚带着它只拿到 10-11 条、地图目视更差。
    #
    # 工具留下了：scratchpad/label_constraints.py 用真值标注约束真假，
    # scratchpad/inspect_voxel.py 按检测器的体素口径查看某处鬼影。想重新引入类似
    # 机制，先用它们证明新判据能命中错回环。

    # Probation: a seed is suspect only if NEW ghost regions implicate the
    # exact segments it connects (global badness is non-monotone in true
    # error: repairs move errors INTO the detectable band).
    if not probation_done and not MINIMAL:
        probation_done = True
        def implicated(region, sid, tid,
                       radius=REPAIR_POLICY['probation_radius_kf']):
            pa, pb = region['suggested_pair']
            return (min(abs(pa - sid), abs(pb - sid)) <= radius
                    and min(abs(pa - tid), abs(pb - tid)) <= radius)
        baseline_pairs = [tuple(r['suggested_pair']) for r in baseline_regions]
        disabled_any = False
        for idx in seed_row_index:
            sid, tid = int(rows[idx][1]), int(rows[idx][2])
            fresh = [r for r in regions if implicated(r, sid, tid)
                     and not any(implicated({'suggested_pair': bp}, sid, tid)
                                 for bp in baseline_pairs)]
            fresh_badness = sum(r['point_count'] * r['separation_m'] for r in fresh)
            if (len(fresh) >= REPAIR_POLICY['probation_min_regions']
                    and fresh_badness > REPAIR_POLICY['probation_min_badness']):
                rows[idx][0] = '0'
                disabled_any = True
                append_proposal_audit(
                    source='stage0_probation', round_id=round_id,
                    source_id=sid, target_id=tid,
                    decision='retracted',
                    reason='new segment-attributed map inconsistencies appeared',
                    original_proposal_id=row_proposal_ids[idx],
                    implicated_region_count=len(fresh),
                    implicated_badness=float(fresh_badness),
                )
                log(f'[Probation] seed {tid}->{sid} SUSPECT: {len(fresh)} new ghost '
                    f'regions at its segments (badness {fresh_badness:.0f}); disabled')
        if disabled_any:
            continue
        log('[Probation] all seeds passed (segment-attributed check)')

    # ``out`` still contains only Initial Loop Search constraints at this point; map
    # proposals have not been tried yet. Preserve it as the exact audited
    # baseline used by C1, including any prior consistency/probation rollback.
    if stage0_audited_out is None:
        stage0_audited_out = out

    if STAGE0_AUDIT_ONLY:
        log('== Initial-loop audit-only control complete; map proposal channel disabled ==')
        break

    if not regions:
        log('== CONVERGED: map clean ==')
        break

    current_traj = load_tum_trajectory(balm_out / 'optimized_poses_tum.txt')
    workspace = RegistrationWorkspace(KF_DIR, current_traj)
    added = 0
    if MODE == 'radius':
        regions = []
        P = current_traj.positions_xyz
        idx = np.arange(len(P))
        for i in range(len(P)):
            d = np.linalg.norm(P - P[i], axis=1)
            far = np.abs(idx - i) > 50
            cand = np.where(far & (d < 5.0))[0]
            if cand.size:
                j = int(cand[np.argmin(d[cand])])
                regions.append({'suggested_pair': [max(i, j), min(i, j)],
                                'separation_m': 0.0, 'normal': [0, 0, 0]})
        regions = regions[:400]
    elif MODE == 'scloop':
        sc_props = find_seed_candidates(
            descriptors, masks, max_seeds=50, max_distance=0.30,
            min_index_gap=50, segment_radius=30,
            channel_weights=CHANNEL_WEIGHTS, num_rings=config.num_rings,
            min_joint_rings=config.min_joint_rings,
            retrieval_height_offset=config.retrieval_height_offset,
            sector_support_exponent=config.sector_support_exponent)
        regions = [{'suggested_pair': [c.source_id, c.target_id],
                    'separation_m': 0.0, 'normal': [0, 0, 0]} for c in sc_props]
    STATS['proposed'] += len(regions)
    for region_index, region in enumerate(regions, 1):
        a, b = region['suggested_pair']
        sid, tid = str(max(a, b)), str(min(a, b))
        proposal_source = (
            'map_inconsistency' if MODE == 'ghost'
            else 'radius' if MODE == 'radius'
            else 'descriptor_round'
        )
        region_id = (
            f'round{round_id:02d}_region{region_index:04d}'
            if MODE == 'ghost' else None
        )
        # 哪一层要往法向的哪一边挪，检测器其实知道：offsets = centered @ normal 是
        # 带符号投影，_two_means_1d 分出 center_low < center_high，而
        # suggested_pair[0] 一定取自低层、[1] 一定取自高层（balm.py:778-781）。
        # 这里按关键帧编号 max/min 重排，把层次归属丢掉了，于是只能 +-sep 各试一次。
        # 恢复它：源若来自低层就该 +sep·normal，来自高层就 -sep·normal。
        sep_sign = 1.0 if int(sid) == int(a) else -1.0
        if (sid, tid) in seen_pairs or sid == tid:
            append_proposal_audit(
                source=proposal_source, round_id=round_id,
                source_id=int(sid), target_id=int(tid), region_id=region_id,
                decision='rejected', reason='duplicate or self-pair',
            )
            continue
        near_dup = any(abs(int(sid) - int(row[1])) <= NEAR_DUP and
                       abs(int(tid) - int(row[2])) <= NEAR_DUP
                       for row in rows)
        if near_dup:
            append_proposal_audit(
                source=proposal_source, round_id=round_id,
                source_id=int(sid), target_id=int(tid), region_id=region_id,
                decision='rejected', reason='near-duplicate active constraint',
            )
            continue
        # Informed cascade: identity, multi-peak SC-yaw priors (works even when
        # retrieval distance is weak — the pair is KNOWN), the ghost's own
        # translation prior (separation along the surface normal, applied to
        # the first two hypotheses), and one collocation seed per yaw peak.
        #
        # Do NOT trim this list from the logged winners. Counting which seed
        # wins says the +-separation priors take 29 of 49 round constraints,
        # identity 16, SC-yaw 4 and collocation 0, which reads like an argument
        # for keeping three of them. Measured instead of assumed, on the four
        # sequences whose ghost rounds accept anything: trimming to
        # identity+-separation drops MCD ntu_day_01 from 23 constraints to 4 and
        # its ATE from 2.77 m back to 3.52 m, and mapping_big from 26 to 17.
        # Two effects the per-seed tally cannot see -- the winner is only the
        # lowest-residual seed among those that PASSED, and rounds compound, so
        # a constraint missed in round 1 leaves the map coarse enough that
        # round 2 diagnoses fewer ghosts, and the session collapses from six
        # rounds to three.
        deltas = [np.eye(4)]
        peaks = pair_yaw_peaks(
            descriptors, masks, int(sid), int(tid),
            CHANNEL_WEIGHTS, config.num_rings,
            min_joint_rings=config.min_joint_rings,
            retrieval_height_offset=config.retrieval_height_offset,
            sector_support_exponent=config.sector_support_exponent,
        )
        sc_dist, sc_yaw = peaks[0]
        Tn = current_traj.transforms_world_sensor
        relative = np.linalg.inv(Tn[int(tid)]) @ Tn[int(sid)]
        colloc_deltas = []
        for _, yaw_deg in (peaks[:1] if ROUND_SINGLE_PEAK else peaks):
            desired = colloc_desired_rotation(int(sid), int(tid), yaw_deg)
            if abs(yaw_deg) > 3.0:
                sc_delta = np.eye(4)
                sc_delta[:3, :3] = relative[:3, :3].T @ desired
                deltas.append(sc_delta)
            desired4 = np.eye(4)
            desired4[:3, :3] = desired
            colloc_deltas.append(
                np.linalg.inv(Tn[int(sid)]) @ Tn[int(tid)] @ desired4)
        normal = (np.array(region.get('normal', [0.0, 0.0, 0.0]))
                  if SEP_PRIOR else np.zeros(3))
        sep = float(region.get('separation_m', 0.0))
        if np.linalg.norm(normal) > 0.5 and sep > 0.0:
            R_src = Tn[int(sid)][:3, :3]
            local_dir = R_src.T @ normal
            signs = (sep_sign,) if SEP_SIGNED else (1.0, -1.0)
            for base in list(deltas)[:2]:
                for sign in signs:
                    d = base.copy()
                    d[:3, 3] = d[:3, 3] + sign * sep * local_dir
                    deltas.append(d)
        # 共位假设只留给初始回环搜索。候选来自漂移地图上的地点识别，两端在当前估计里
        # 可能相隔几十米，里程计相对位姿没有参考价值，必须把源直接放到目标处；
        # 而轮次提案来自已经基本正确的地图上诊断出的鬼影，两层只差 8-35 cm，
        # 真解就在 identity 附近，共位反而把源扔到地图另一头。
        # 统计也一致：49 条轮次约束里共位赢 0 条。
        # 注意这和 2026-08-04 测过的"砍成 identity+-分离量"不是一回事——那次连
        # SC-yaw 一起砍了，而 +-分离量是加在 deltas[:2]（identity 和第一个 SC-yaw）
        # 上的，所以那次把"SC-yaw 附近的 +-分离量"也一并删掉了，ntu 从 23 条掉到 4 条。
        if COLLOC_IN_ROUNDS:
            deltas.extend(colloc_deltas)
        log(f'[Round{round_id}] pair {tid}->{sid}: sc_yaw={sc_yaw:.0f}deg '
            f'(d={sc_dist:.2f}), {len(deltas)} seeds')
        r = gicp_gate(int(sid), int(tid), deltas, f'Round{round_id}')
        if r is None:
            attempted = _LAST_GATE_RESULT.get((int(sid), int(tid)))
            factor_reason = _LAST_FACTOR_REASON.pop((int(sid), int(tid)), None)
            gravity_reason = _LAST_GRAVITY_REASON.pop(
                (int(sid), int(tid)), None
            )
            append_proposal_audit(
                source=proposal_source, round_id=round_id,
                source_id=int(sid), target_id=int(tid), region_id=region_id,
                decision='rejected', reason=(
                    f'oriented factor gate failed: {factor_reason}'
                    if factor_reason else (
                        f'gravity consistency gate failed: {gravity_reason}'
                        if gravity_reason else
                        'GICP or odometry-budget gate failed'
                    )
                ),
                result=attempted,
                detector={
                    'typical_separation_m': region.get('separation_typical_m'),
                    'point_count': region.get('point_count'),
                    'voxel_count': region.get('voxel_count'),
                    'layer_overlap_ratio': region.get('layer_overlap_ratio'),
                },
            )
            continue

        # Map proposals must prove that their own immutable evidence responds
        # to the proposed graph factor.  This is a candidate-local causal test,
        # not the old global-badness heuristic.
        evidence = None
        before_separation = after_separation = None
        effect = None
        candidate_residual = None
        trial_worst = None
        candidate_nis = None
        candidate_consistency_score = None
        trial_worst_consistency_score = None
        trial_dir = None
        if MODE == 'ghost':
            try:
                evidence = freeze_region_evidence(
                    region, _PREP_CLOUDS,
                    current_traj.transforms_world_sensor,
                    root_voxel_size=balm_params.root_voxel_size,
                    max_observation_range=GHOST_RANGE,
                )
                before_separation = evidence.reference_separation_m
            except Exception as exc:
                append_proposal_audit(
                    source=proposal_source, round_id=round_id,
                    source_id=int(sid), target_id=int(tid), region_id=region_id,
                    decision='rejected', reason=f'fixed evidence freeze failed: {exc}',
                    result=r,
                )
                continue

            pid = proposal_id(
                RUN_ID, proposal_source, round_id, int(sid), int(tid), region_id
            )
            trial_dir = out / 'proposal_trials' / pid
            trial = trial_graph_audit(
                rows, constraint_row(r), trial_dir,
                out / 'optimized_poses_tum.txt',
            )
            if trial['completed'].returncode != 0:
                append_proposal_audit(
                    source=proposal_source, round_id=round_id,
                    source_id=int(sid), target_id=int(tid), region_id=region_id,
                    decision='rejected', reason='trial PGO failed', result=r,
                )
                continue
            evidence.save(trial_dir / 'fixed_evidence.npz')
            trial_traj = load_tum_trajectory(
                trial_dir / 'optimized_poses_tum.txt'
            )
            after_separation = measure_frozen_evidence(
                evidence, trial_traj.transforms_world_sensor
            )
            effect = evaluate_causal_effect(before_separation, after_separation)
            candidate_residual = trial['candidate_residual_m']
            trial_worst = trial['worst_factor_residual_m']
            candidate_nis = trial['candidate_factor_nis']
            candidate_consistency_score = trial['candidate_consistency_score']
            trial_worst_consistency_score = (
                trial['worst_factor_consistency_score']
            )
            graph_ok = trial['graph_ok']
            if not effect.retained or not graph_ok:
                reason = (
                    effect.reason if not effect.retained
                    else 'trial factor triggered global residual audit'
                )
                append_proposal_audit(
                    source=proposal_source, round_id=round_id,
                    source_id=int(sid), target_id=int(tid), region_id=region_id,
                    decision='retracted', reason=reason, result=r,
                    fixed_evidence={
                        'path': str((trial_dir / 'fixed_evidence.npz').resolve()),
                        'point_count': int(len(evidence.local_xyz)),
                        'before_separation_m': float(before_separation),
                        'after_separation_m': float(after_separation),
                        'absolute_drop_m': float(effect.absolute_drop_m),
                        'relative_drop': float(effect.relative_drop),
                    },
                    factor_residual_m=candidate_residual,
                    worst_factor_residual_m=trial_worst,
                    factor_nis=candidate_nis,
                    factor_consistency_score=candidate_consistency_score,
                    worst_factor_consistency_score=(
                        trial_worst_consistency_score),
                )
                log(f'[causal-audit] {tid}->{sid}: {reason} '
                    f'({before_separation*100:.1f}->{after_separation*100:.1f} cm)')
                continue

        rows.append(constraint_row(r))
        seen_pairs.add((sid, tid))
        pid = append_proposal_audit(
            source=proposal_source, round_id=round_id,
            source_id=int(sid), target_id=int(tid), region_id=region_id,
            decision='retained',
            reason=(effect.reason if effect is not None
                    else 'passed shared pre-solve admission'),
            result=r,
            fixed_evidence=(None if evidence is None else {
                'path': str((trial_dir / 'fixed_evidence.npz').resolve()),
                'point_count': int(len(evidence.local_xyz)),
                'before_separation_m': float(before_separation),
                'after_separation_m': float(after_separation),
                'absolute_drop_m': float(effect.absolute_drop_m),
                'relative_drop': float(effect.relative_drop),
            }),
            factor_residual_m=candidate_residual,
            worst_factor_residual_m=trial_worst,
            factor_nis=candidate_nis,
            factor_consistency_score=candidate_consistency_score,
            worst_factor_consistency_score=trial_worst_consistency_score,
        )
        row_proposal_ids.append(pid)
        added += 1
        STATS['accepted'] += 1
        if PER_ROUND_CAP and added >= PER_ROUND_CAP:
            log(f'[Round{round_id}] per-round cap {PER_ROUND_CAP} reached')
            break
    if added == 0:
        log('== CONVERGED: no addable proposals remain ==')
        break
else:
    log('== stopped at round limit ==')

# Legacy proposal modes may accept their last constraint immediately before a
# stop, so materialize that exact final active set. Production has already
# solved the audited set and either kept its PGO result (audit-only) or carried
# it into the one final BALM directory; re-solving here would discard BALM.
if MODE != 'production':
    final_out = SESSION / 'manual_loop_runs' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_auto_final'
    )
    final_solve = solve_constraint_rows(rows, final_out)
    if final_solve.returncode != 0:
        log(final_solve.stdout[-1500:]); log(final_solve.stderr[-1500:])
        raise SystemExit('final optimizer materialization failed')
    out = final_out

for index, row in enumerate(rows):
    active = row[0] == '1'
    append_proposal_audit(
        source='proposal_finalization', round_id=round_id,
        source_id=int(row[1]), target_id=int(row[2]),
        decision='retained_final' if active else 'inactive_final',
        reason=(
            'production audited initial-loop active set'
            if MODE == 'production'
            else 'terminal active-set materialization'
        ),
        original_proposal_id=row_proposal_ids[index],
        active=active,
    )

ledger_sha256 = hashlib.sha256(LEDGER_PATH.read_bytes()).hexdigest()
if MODE == 'production':
    log('== terminal policy: Initial Loop Search -> Audited PGO -> one Final Map Refinement ==')
else:
    log('== terminal policy: no new retained loop; final active set materialized ==')

log(f"SUMMARY mode={MODE} proposed={STATS['proposed']} gicp_pairs={STATS['gicp_pairs']} "
    f"accepted={STATS['accepted']} constraints={sum(r[0]=='1' for r in rows)}")
write_repair_summary(
    'complete',
    rounds_completed=round_id,
    active_constraint_count=sum(row[0] == '1' for row in rows),
    termination_policy=(
        'stage0_audit_only' if MODE == 'production' and STAGE0_AUDIT_ONLY
        else 'single_final_double_sided_balm' if MODE == 'production'
        else 'new_accepted_loop_only'
    ),
    balm_passes=(1 if production_balm_completed else 0),
    balm_adopted=production_balm_adopted,
    terminal_output_dir=str(out.resolve()),
    stage0_audited_output_dir=(
        None if stage0_audited_out is None
        else str(stage0_audited_out.resolve())
    ),
    proposal_ledger_sha256=ledger_sha256,
)
write_result_contract(
    out,
    active_constraint_count=sum(row[0] == '1' for row in rows),
    balm_completed=production_balm_completed,
    balm_adopted=production_balm_adopted,
)
log('DONE')
