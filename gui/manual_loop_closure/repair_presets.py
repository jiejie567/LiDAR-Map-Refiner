"""Dependency-free Repair Map environment presets.

Keep policy values here so the GUI, headless evaluator, and fail-closed result
validator cannot silently drift apart.  This module intentionally imports no
Open3D, Qt, or NumPy code; pipeline validation can therefore use system Python.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

from .registration import OFFICE_DEFAULT_VARIANCE_T


# One-click Repair Map pose-graph profile.  Keep this separate from the
# editable expert widgets: a production repair must not change because a wheel
# event happened over a spin box, while manual constraint experiments should
# remain possible.
#
# The profile was selected on Building Day, MCD KTH/NTU and Oxford Spires from
# a frozen GICP proposal set, then checked once (without retuning) on M2DGR
# hall02/hall04.  The held-out change was +1 mm on hall02 and zero on hall04;
# all four development sequences improved.  The deliberately small model is:
# diagonal GICP factors, one global information scale, GNC-TLS, and no
# dataset-specific covariance/correlation knobs.
#
# A final Occam check also compared the tempting ``sigma_r=5 deg, scale=0.03``
# profile.  It reduced prefix-to-prefix ATE oscillation on 58 place-correct
# proposals, but did not dominate on the frozen complete candidate pool: the
# geometric-mean final ATE ratio was 0.6107 versus 0.6060 here.  All four
# datasets improved under this profile and the held-out controls stayed safe,
# so adding a dataset/count-dependent profile switch would buy complexity, not
# a demonstrated accuracy or safety gain.  Geometry-derived anisotropic loop
# information likewise improved Spires but not Building Day.  Keep both as
# explicit ablations rather than hidden production policy.
PRODUCTION_PGO_PROFILE = {
    "name": "gicp_diagonal_gnc_v1",
    "optimize_mode": "gnc_tls",
    "translation_variance_m2": float(OFFICE_DEFAULT_VARIANCE_T[0]),
    "rotation_sigma_deg": 0.5,
    "manual_information_scale": 0.1,
    "correlation_window_keyframes": 0,
    "cluster_information_budget": 0.0,
    "correlation_policy": "pair",
    "cluster_allocation": "equal",
    # This is only a catastrophic roll/pitch sanity check, not an accuracy
    # gate.  The previous 1-degree cutoff removed the long loops that repair
    # MCD NTU.  Ten degrees is intentionally far outside descriptor
    # quantisation and measured gravity noise, and did not bind the frozen
    # development or held-out proposal sets.
    "gravity_sanity_limit_deg": 10.0,
}


def production_loop_variances() -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    """Return the GUI's stored m^2/rad^2 variances for production loops."""
    translation = float(PRODUCTION_PGO_PROFILE["translation_variance_m2"])
    rotation = math.radians(
        float(PRODUCTION_PGO_PROFILE["rotation_sigma_deg"])
    ) ** 2
    return (translation,) * 3, (rotation,) * 3


REPAIR_ENV_PRESETS = {
    "indoor": {
        "voxel": 0.20,
        "max_corr": 2.0,
        "wide_max_corr": 3.0,
        "balm_voxel": 1.0,
        "target_neighbors": 40,
        "target_map_voxel": 0.20,
        "balm_downsample": 0.20,
        "balm_max_range": 80.0,
        # Convergence-driven coarse/fine refinement. Twenty is a runtime cap;
        # each stage stops earlier on pose-update convergence or RMS plateau.
        # The old two-update value was selected with ground truth and is kept
        # only as an explicit ablation, not a deployable stopping rule.
        "balm_iterations": 20,
        "balm_double_sided": True,
        # Scan Context cloud trim only; the ghost diagnosis reads
        # REPAIR_POLICY["ghost_observation_range_m"].
        "observation_range": 20.0,
        # 抑制规则保持"两端都落在 segment_radius 内才算重复"，不要改成"任一端"。
        # 更严的版本实测把真重访杀掉一大半（ntu 14->5、in-house 20->5、sim 6->2），
        # 因为共用一端的候选是合法的：同一处被经过三次会形成 A-B/A-C/B-C 三对，
        # 长走廊的两端也可能各自和别处重访。
        #
        # 名额不再是策略，只是防病态输入的兜底。真正决定发多少张票的是
        # seed_max_distance 加上抑制半径：贪心从最近的取起，每取一个就抑制它
        # 前后 segment_radius 帧，直到描述子距离超过阈值。票数因此是序列的属性
        # ——escalator00 只有 5 处重访就发 5 张，不再硬凑 8 张。
        "seed_max_candidates": 64,
        # 0.32，和室外同一个值。原本保留 0.45，依据是三条小室内序列（156-378 帧）
        # 一个错候选都没有；但那三条代表不了大场景——in-house 2579 帧 / 920 m 的
        # 楼里，0.45 一路取到兜底上限 64 张票才停，而它的距离分布从 0.072 平滑
        # 爬到 0.558、最大跳变只有 0.023，没有可依据的断层。
        # 改成 0.32 后 in-house 降到 20 张，七条序列全部零错候选。
        "seed_max_distance": 0.32,
    },
    "outdoor": {
        "voxel": 0.40,
        "max_corr": 2.0,
        "wide_max_corr": 8.0,
        "balm_voxel": 4.0,
        "target_neighbors": 40,
        "target_map_voxel": 0.40,
        # Match the frozen official-BALM2 comparison and blind holdout.
        "balm_downsample": 0.40,
        "balm_max_range": 80.0,
        # Same convergence cap indoors/outdoors; this is not an IMU/PGO prior.
        "balm_iterations": 20,
        "balm_double_sided": True,
        # Scan Context cloud trim only; see the indoor note.
        "observation_range": 30.0,
        # 8, not the 5 this used to be. Retrieval is recall-first by design --
        # "retrieval only has to surface candidates", with the GICP cascade,
        # the odometry budget, probation and the repair check carrying
        # precision -- so an outdoor cap BELOW the indoor one was backwards for
        # the larger scenes with more revisits, and it was costing real loops.
        #
        # Measured on MCD ntu_day_01. The two constraints that tie the end of
        # the sequence back to its first twenty keyframes -- the ones that fix
        # global drift -- rank #6 (d=0.155) and #8 (d=0.157), just behind a
        # top-5 whose last entry is d=0.152. The distance cap of 0.80 never
        # binds; the COUNT was the whole limiter. An earlier run that did admit
        # them reached 1.31 m ATE where the top-5 run stalls at 3.47 m.
        "seed_max_candidates": 64,
        # 0.32，不是原来的 0.80。距离 = 1 - 相似度（seed_loops.py:239），
        # 0.80 等于"相似度 20% 就算候选"，形同虚设：ntu 上放到 80 个候选，
        # 62 个是错的，真正在停的是那个和序列长度无关的 8 张票。
        #
        # 0.32 是三条室外序列上量出来的分界，真假两类几乎不重叠：
        #     ntu  真 0.121-0.401  错最小 0.400   分界 0.39
        #     kth  真 0.178-0.192  错最小 0.382   分界 0.38
        #     bd   真 0.073-0.306  错最小 0.332   分界 0.33
        # 取最保守的一侧，三条上都零假阳且真重访一个不丢（bd 最高 0.306）。
        "seed_max_distance": 0.32,
    },
}


# The geometric acceptance gate, shared for the same reason the presets are:
# the GUI and the headless evaluator each carried their own copy, and on
# 2026-08-03 they silently disagreed. One in-house session repaired twice from
# the same odometry gave 18 constraints in 4 rounds under an overlap floor of
# 0.6 and 36 in 10 rounds under 0.5 -- the first two rounds identical, then
# diverging as soon as a candidate landed between the two thresholds. The
# resulting maps were equivalent (117 versus 119 residual ghost regions against
# odometry's 147), so the extra constraints were redundant rather than wrong,
# but "same data, same buttons, two answers" is not a property this pipeline
# can afford.
#
# Two independent criteria, as in the C++ relocaliser this was ported from
# (prior_icp.cpp): how much of the scan found a partner at all, and how far
# those partners sit. 0.5 is the overlap floor already validated there. The
# residual bound is a fraction of the voxel edge rather than a fixed distance,
# because a correct registration cannot beat the quantisation its own
# downsampling imposes -- at the 0.2 m indoor voxel that reproduces the 0.18 m
# the headless pipeline has always used, and at 0.4 m outdoor it becomes 0.36
# instead of staying at a floor that would reject every genuine outdoor loop.
REPAIR_GATE = {
    # 0.6, not the C++ relocaliser's 0.5. Both were run end to end on the
    # 2,579-keyframe in-house session from the same odometry: 0.6 converges in
    # four rounds on eighteen constraints, 0.5 keeps going for ten and reaches
    # thirty-six, and the operator judged the second map visibly wrong. The
    # looser floor does not merely admit more loops -- selection inside a tier
    # takes the smallest residual, so a barely-overlapping registration that
    # happens to score a little lower displaces the one that should have won.
    "min_overlap": 0.6,
    # 1.1, not the 0.9 this was. 0.9 x a 0.2 m indoor voxel puts the bound at
    # 0.18 m, and this file's own calibration note records TRUE revisit pairs at
    # 0.15-0.22 m RMSE -- the bound ran straight through the middle of the
    # distribution it was supposed to sit above, rejecting real loops rather
    # than false ones.
    #
    # Measured end to end on FusionPortable escalator00 (indoor, native defect,
    # Leica-grade truth, odometry 24.2 cm):
    #     ratio 0.9 -> bound 0.18 -> 2 constraints -> 18.8 cm
    #     ratio 1.1 -> bound 0.22 -> 5 constraints -> 14.9 cm
    #     ratio 1.3 -> bound 0.26 -> 5 constraints -> 15.4 cm
    #     ratio 1.6 -> bound 0.32 -> 5 constraints -> 14.7 cm
    # A plateau from 1.1 to 1.6 and a cliff below it: 0.9 was the only outlier,
    # which is what a threshold sitting inside the true-loop population looks
    # like. On MCD ntu two candidates at fitness 0.999 missed the outdoor bound
    # by 3 mm (0.363 against 0.360) while the nearest false loop scored 0.543 --
    # a 17 cm gap between the populations, with the bound drawn inside the true
    # one rather than in the gap.
    #
    # 1.1 and not the middle of the plateau because the cascade stops at the
    # first tier that passes, so a looser bound lets an EARLIER, coarser tier
    # win with a worse measurement. Measured on the same sequence: 76->136
    # commits rmse 0.212 (wide) at 1.1 but 0.244 (near) at 1.3, and 21->67
    # commits 0.185 (anneal) at 1.1 but 0.297 (wide) at 1.6. The smallest
    # sufficient bound keeps the best measurement.
    "rmse_voxel_ratio": 1.1,
    "rmse_floor": 0.15,          # for voxels small enough to beat the ratio
}

# Kept overridable: the C++ path this was ported from uses 0.5, and comparing
# the two is how the divergence above was found.
# The GUI applied its environment preset only on a dropdown CHANGE, so a
# session repaired without touching it ran at the widget default of 0.0 --
# which the registration floors to 0.05, four times finer than indoor's
# 0.20. Overridable so the two can be compared head to head.
_VOXEL_OVERRIDE = os.environ.get("GHOSTLOOP_VOXEL")
if _VOXEL_OVERRIDE:
    try:
        for _p in REPAIR_ENV_PRESETS.values():
            _p["voxel"] = float(_VOXEL_OVERRIDE)
    except ValueError:
        pass

# The residual bound, for the same reason: this file's own calibration note
# records true revisit pairs at 0.15-0.22 m RMSE, and 0.9 x a 0.2 m voxel puts
# the bound at 0.18 -- straight through the middle of that distribution.
_RMSE_RATIO_OVERRIDE = os.environ.get("GHOSTLOOP_RMSE_RATIO")
if _RMSE_RATIO_OVERRIDE:
    try:
        REPAIR_GATE["rmse_voxel_ratio"] = float(_RMSE_RATIO_OVERRIDE)
    except ValueError:
        pass

# The ICP target map's own resolution, overridable for the same reason: on
# 2026-08-02 the defaults went from 100 neighbours at a 0.1 m voxel to 40 at
# 0.2, and the outdoor configs additionally started passing 0.4 explicitly
# where they had passed nothing (i.e. 0.1) before. Every registration's
# achievable residual is bounded below by the target's own quantisation, so
# this silently reset the precision of every constraint the pipeline measures.
_TARGET_VOXEL_OVERRIDE = os.environ.get("GHOSTLOOP_TARGET_MAP_VOXEL")
if _TARGET_VOXEL_OVERRIDE:
    try:
        for _p in REPAIR_ENV_PRESETS.values():
            _p["target_map_voxel"] = float(_TARGET_VOXEL_OVERRIDE)
    except ValueError:
        pass

_TARGET_NEIGHBORS_OVERRIDE = os.environ.get("GHOSTLOOP_TARGET_NEIGHBORS")
if _TARGET_NEIGHBORS_OVERRIDE:
    try:
        for _p in REPAIR_ENV_PRESETS.values():
            _p["target_neighbors"] = int(_TARGET_NEIGHBORS_OVERRIDE)
    except ValueError:
        pass

_OVERLAP_OVERRIDE = os.environ.get("GHOSTLOOP_MIN_OVERLAP")
if _OVERLAP_OVERRIDE:
    try:
        REPAIR_GATE["min_overlap"] = float(_OVERLAP_OVERRIDE)
    except ValueError:
        pass


# A proposal is a near duplicate when an existing constraint already joins
# segments within this many keyframes, so one loop is not added a dozen times
# over. The window looks blunt -- it vetoes on POSITION and never asks whether
# the constraint already there is any good -- and on FusionPortable escalator00
# it does suppress a proposal the diagnosis keeps asking for (a 35 cm ghost
# between kf 1-31 and kf 126-154, covered by constraint 6<->149 whose
# re-measured residual is 0.144 m against 0.008-0.013 for every other
# constraint in that run).
#
# Measured anyway, because "looks blunt" is not evidence: shrinking the window
# to 5 admits that proposal and 20 others, and the ATE gets WORSE, 28.9 -> 31.5
# cm (odometry 34.7). The duplicate factors over-weight their loop against the
# rest of the graph by more than the better measurement is worth. 30 stays.
NEAR_DUPLICATE_KEYFRAMES = int(os.environ.get("GHOSTLOOP_NEAR_DUP", "30"))


# Numeric policy for the safeguards that run DURING a repair, as opposed to the
# per-pair acceptance gate above. Here for one reason: every one of these used
# to exist twice, once in the GUI and once in the headless evaluator, and a pair
# that drifted apart cost a day of reproducing a run that could not be
# reproduced. A constant with two homes has no value; it has two values.
REPAIR_POLICY = {
    # 3 sigma of the loop factor's own noise. A constraint whose post-solve
    # residual exceeds this is not merely imperfect -- the graph could not
    # satisfy it within the uncertainty it was added with.
    #
    # Derived from OFFICE_DEFAULT_VARIANCE_T rather than repeating its value:
    # the two were separate literals (both 0.1) until 2026-08-06, so loosening
    # the factor noise would silently have left this gate at the old scale.
    # A constant with two homes has no value; it has two values.
    "residual_limit_m": 3.0 * math.sqrt(OFFICE_DEFAULT_VARIANCE_T[0]),
    # 里程计一致性预算：一条回环的修正量不能超过里程计在两端之间那条链上
    # 可信地漂移出来的量。自相似街道会以很高的置信度配到错误的偏移上（fitness
    # 和 rmse 都好看），只有里程计链能揭穿这个矛盾。
    #
    # 比率原为 1.5%，那是"现代 LIO 漂移率的两倍"这个关于**前端**的假设，不是
    # 数据的性质，也不迁移：MCD kth_night_01 漂了路径长度的 5.4%，1.5% 的预算会把
    # 它两条真实重访全部拒掉（73.8 m 的修正对 7.4 m 的预算）。所以预算放宽，让
    # 优化后的因子残差（一个测量，不是猜测）去做最终判决。
    #
    # 2026-08-06 从 GUI 和无头各自的字面量收拢到这里。此前 GUI 写
    # `(1.0 if outdoor else 0.5) + 0.10 * chain`，无头写 BUDGET_BASE/BUDGET_RATE，
    # 两处独立 —— 同一个判据有两个家就等于有两个值。
    "budget_base_indoor_m": 0.5,
    "budget_base_outdoor_m": 1.0,
    "budget_rate": 0.10,

    # Plane refinement should move poses centimetres. Metre-scale motion means
    # the BA geometry is degenerate (no horizontal support in narrow-FOV data,
    # say) and its output must be dropped for the poses that went in.
    "balm_sanity_limit_m": 0.5,
    # Translation alone cannot detect a planar BA that spins poses in place.
    # This is deliberately a catastrophic mean-motion guard, not a gravity or
    # IMU prior inside BALM.  Normal terminal refinement is sub-degree (0.39°
    # mean on the frozen NTU production run); a ten-degree session-wide mean
    # is already far outside the intended role of a final local refinement.
    "balm_sanity_limit_rotation_deg": 10.0,
    # Below this map inconsistency there is nothing for BA to merge, and motion
    # no defect asked for is pure risk: on Hilti exp14, BA against zero
    # diagnosed ghosts took a 12 cm session to 40 cm.
    "balm_min_badness": 1.0,
    # Seed probation, judged per segment rather than globally -- global badness
    # is not monotone in true error, because a repair moves error INTO the
    # detectable band. A seed is suspect only when NEW ghosts implicate both of
    # its own endpoints: within this many keyframes, at least this many of them,
    # jointly at least this heavy. One region is within the noise of the
    # residual seam any repair leaves.
    "probation_radius_kf": 40,
    "probation_min_regions": 2,
    "probation_min_badness": 1000.0,
    # A ghost separation wider than this is not a ghost the seed prior can
    # describe, and the support window for deciding a seed is warranted.
    "ghost_max_separation_m": 0.35,
    "seed_support_tolerance_kf": 60,
    # How far from its own sensor an observation may be and still count toward a
    # ghost diagnosis. NOT per environment, and not the same job as the preset's
    # `observation_range` below, which trims the clouds the Scan Context stack is
    # built from -- that one wants to be WIDE (a bigger scene is a more
    # distinctive descriptor) while this one wants to be TIGHT.
    #
    # Tight, because pose-error ghosting is range-independent but angular noise
    # misplaces returns linearly with range: far observations of one surface
    # masquerade as two layers. At 0.2 deg resolution the noise smear reaches the
    # 0.08 m lower bound of the ghost band at about 23 m, so 20 sits just inside
    # the distance where a "ghost" can still only be a real one.
    #
    # Measured rather than reasoned, on one indoor and one outdoor session, by
    # surveying the same trajectory at 20 and at 30. Widening does not merely add
    # detections, it DESTROYS them: the extra smeared returns fill the gap
    # between the two layers, so the offset distribution stops being bimodal and
    # the layer-tightness test rejects it. Indoor 127 regions -> 156, but 36 of
    # the original 127 vanish and 18 of the 62 newcomers hug the 0.08 m floor.
    # Outdoor 4 -> 9 with total severity going DOWN, 520 -> 456: the three that
    # vanished were the heavy ones.
    "ghost_observation_range_m": 20.0,
}


def repair_environment_preset(environment: str) -> dict:
    """Return an isolated preset mapping for ``environment``."""
    try:
        return deepcopy(REPAIR_ENV_PRESETS[environment])
    except KeyError as exc:
        raise ValueError(f"unsupported Repair environment: {environment}") from exc


def repair_gate(voxel_size: float) -> dict:
    """Acceptance thresholds for a registration at ``voxel_size``.

    Returns ``{"min_fitness": ..., "max_rmse": ...}``. Both entry points must
    read this rather than keep their own constants.
    """
    ratio = REPAIR_GATE["rmse_voxel_ratio"] * float(voxel_size)
    return {
        "min_fitness": REPAIR_GATE["min_overlap"],
        "max_rmse": max(REPAIR_GATE["rmse_floor"], ratio),
    }
