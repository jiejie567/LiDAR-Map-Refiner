#!/usr/bin/env python3
"""我们独有的那些约束，在描述子排序里到底排第几名。

这条实验回答的是"换个更强的检索器会不会就够了"。论文的论断是：那些回环之所以
找不到，不是相似度阈值卡掉的，而是两端互相都不进对方的近邻表 —— 这是 top-k
检索这个**形式**的盲区，与用哪个描述子无关。要证成它，就得把每条约束在排序里
的名次量出来。

**排序必须用流水线自己检索时用的那个距离。** `find_seed_candidates`
(seed_loops.py) 调的是 `_yaw_distances(...)[0]`：先粗搜 yaw 再在局部窗口精搜，
即"对齐朝向之后的最小距离"。早先这个脚本用的是 `_descriptor_distance`，那是
**固定 yaw** 下的单次距离；反向经过或换朝向的回环在它下面距离必然很大，于是名次
被系统性地夸大，恰好夸大在鬼影诊断最擅长的那一类上。

同理，描述子栈必须带上重力 sidecar（流水线在
`gravity_canonicalization_enable` 为真时会传），近邻表的时间间隔排除必须与
与初始回环搜索一致（`min(150, N//4)`，不是写死的 60）。

脚本启动时会把自己算出的距离与流水线日志里记的那几个值对一遍，对不上就报错——
这是唯一能保证"量的是同一个东西"的办法。

用法：retrieval_rank.py <会话目录> <环境> <约束对文件> <该会话的运行日志>
      日志不是可选的：脚本用它对账，确认自己算的距离与流水线检索时算的是同一个。
      约束对文件每行一个 "a->b"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / 'gui'))
from manual_loop_closure import RegistrationWorkspace, load_tum_trajectory  # noqa: E402
from manual_loop_closure.python_optimizer.balm import prepare_local_cloud  # noqa: E402
from manual_loop_closure.repair_presets import repair_environment_preset  # noqa: E402
from manual_loop_closure.scan_context_io import (  # noqa: E402
    load_scan_context_config, load_scan_context_gravity,
)
from manual_loop_closure.python_optimizer.balm import BalmParams  # noqa: E402
from manual_loop_closure.seed_loops import (  # noqa: E402
    build_descriptor_and_prepared_stack,
    descriptor_env_setup,
    _yaw_distances,
)


def main() -> None:
    session = Path(sys.argv[1]).expanduser()
    env = sys.argv[2]
    pairs = [tuple(int(v) for v in ln.replace('->', ' ').split())
             for ln in Path(sys.argv[3]).read_text().split('\n') if ln.strip()]
    log_path = Path(sys.argv[4]).expanduser()

    traj = load_tum_trajectory(session / 'optimized_poses_tum.txt')
    ws = RegistrationWorkspace(session / 'key_point_frame', traj)
    cfg, weights = descriptor_env_setup(load_scan_context_config(session), env)
    preset = repair_environment_preset(env)
    dparams = BalmParams(root_voxel_size=preset['balm_voxel'],
                         downsample_leaf=preset['balm_downsample'],
                         max_range=preset['balm_max_range'],
                         max_iterations=preset['balm_iterations'],
                         double_sided_enable=preset['balm_double_sided'])
    n = traj.size

    gravity = None
    if cfg.gravity_canonicalization_enable:
        try:
            gravity = load_scan_context_gravity(session, traj)
        except Exception as exc:
            print(f'  重力 sidecar 不可用（{exc}），与流水线不一致，结果不可用')
    desc, masks, _ = build_descriptor_and_prepared_stack(
        ws.load_local_points_uncached,
        lambda pts: prepare_local_cloud(pts, dparams),
        n, cfg, gravity, log_fn=lambda *_: None)
    print(f'  描述子栈 {n} 帧，环境 {env}，重力规范化 {"开" if gravity is not None else "关"}')

    kw = dict(channel_weights=weights, num_rings=cfg.num_rings,
              min_joint_rings=cfg.min_joint_rings,
              retrieval_height_offset=cfg.retrieval_height_offset,
              sector_support_exponent=cfg.sector_support_exponent)

    def dist(a: int, b: int) -> float:
        s = _yaw_distances(desc[a], masks[a], desc[b], masks[b], **kw)
        return s[0][0] if s else np.inf

    # 自检：与流水线日志里记下的距离对一遍。对不上就说明量的不是同一个东西。
    checked = bad = 0
    for ln in log_path.read_text().splitlines():
        m = re.search(r'\[Round\d+\] pair (\d+)->(\d+): sc_yaw=.*\(d=([\d.]+)\)', ln)
        if not m:
            continue
        a, b, want = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if max(a, b) >= n:
            continue
        got = dist(a, b)
        checked += 1
        if abs(got - want) > 0.006:      # 日志只记两位小数
            bad += 1
            print(f'    自检不符 {a}->{b}: 日志 {want:.2f} vs 本脚本 {got:.4f}')
    print(f'  自检：{checked - bad}/{checked} 条与流水线日志一致')
    # 一条都没对上说明日志里没有可比的行 —— 那等于没自检，不能放行。
    if not checked:
        raise SystemExit(f'{log_path} 里没有可对账的 sc_yaw 行，无法确认距离口径')
    if bad:
        raise SystemExit('排序用的距离与流水线不一致，结果不可用')

    MIN_GAP = min(150, n // 4)      # 与初始回环搜索一致

    def ranks_from(a: int) -> np.ndarray:
        d = np.full(n, np.inf)
        for b in range(n):
            if abs(a - b) >= MIN_GAP:
                d[b] = dist(a, b)
        return np.argsort(d)

    cache: dict[int, np.ndarray] = {}
    print(f'\n{"约束":>14} {"b 在 a 的名次":>13} {"a 在 b 的名次":>13} {"最好名次":>9}')
    best_all = []
    for a, b in pairs:
        if max(a, b) >= n:
            continue
        for x in (a, b):
            if x not in cache:
                cache[x] = ranks_from(x)
        ra = int(np.where(cache[a] == b)[0][0]) + 1
        rb = int(np.where(cache[b] == a)[0][0]) + 1
        best = min(ra, rb)
        best_all.append(best)
        print(f'{a:>6}->{b:<7} {ra:>13} {rb:>13} {best:>9}')
    if best_all:
        arr = np.array(best_all)
        print(f'\n  最好名次：中位 {int(np.median(arr))}，最小 {arr.min()}，最大 {arr.max()}')
        for k in (1, 5, 10, 25, 50, 100, 200):
            print(f'    top-{k:<4} 能覆盖 {int((arr <= k).sum())}/{len(arr)} 条')


if __name__ == '__main__':
    main()
