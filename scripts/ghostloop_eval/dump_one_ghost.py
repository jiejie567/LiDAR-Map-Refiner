#!/usr/bin/env python3
"""转储单个鬼影体素：检测器实际看到的点、它的分层标签、贡献每层的关键帧。

teaser 图要展示"检测出来的鬼影长什么样"，就必须画检测器真正用的那批点。
逐字复现 detect_ghost_regions 的前处理，其中最关键的一步是 20 m 观测距离
过滤 —— 位姿误差造成的重影与距离无关，但角噪声造成的错位随距离线性增长，
所以远处观测会伪装成分层。不加这个过滤，同一处会糊成单层。

体素化用 root_voxel_size 的平铺网格（没有自适应细分），每个体素内 PCA 求法向、
沿法向做一维二均值分层 —— 这就是"鬼影"的定义。

用法：dump_one_ghost.py <tum> <x> <y> <z> <out.npz> [<会话目录>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / 'gui'))
sys.path.insert(0, str(TOOL / 'gui/manual_loop_closure/python_optimizer'))
from manual_loop_closure import RegistrationWorkspace, load_tum_trajectory  # noqa: E402
from balm import BalmParams, _two_means_1d  # noqa: E402

KF_DEFAULT = Path.home() / 'icra2027_runtime/manual_loop/original/key_point_frame'
MAX_RANGE = 20.0          # detect_ghost_regions 的 max_observation_range 默认值


def main() -> None:
    tum = Path(sys.argv[1]).expanduser()
    target = np.array([float(v) for v in sys.argv[2:5]])
    out = Path(sys.argv[5]).expanduser()
    session = Path(sys.argv[6]).expanduser() if len(sys.argv) > 6 else None

    kf = (session / 'key_point_frame') if session else KF_DEFAULT
    root = 1.0
    if session and (session / 'auto_repair_summary.json').exists():
        root = json.loads((session / 'auto_repair_summary.json').read_text()) \
            .get('balm', {}).get('root_voxel_size', 1.0)

    traj = load_tum_trajectory(tum)
    ws = RegistrationWorkspace(kf, traj)
    R = traj.transforms_world_sensor[:, :3, :3]
    t = traj.transforms_world_sensor[:, :3, 3]

    # 只取目标附近的关键帧，其余对该体素不可能有贡献（20 m 上限）
    keep = np.where(np.linalg.norm(traj.positions_xyz - target, axis=1)
                    < MAX_RANGE + 5.0)[0]
    pts, pose = [], []
    for i in keep:
        w = ws.load_local_points(int(i)) @ R[i].T + t[i]
        near = np.einsum('ij,ij->i', w - t[i], w - t[i]) <= MAX_RANGE ** 2
        w = w[near]
        if not len(w):
            continue
        pts.append(w)
        pose.append(np.full(len(w), i))
    world = np.vstack(pts)
    all_pose = np.concatenate(pose)

    key = np.floor(target / root).astype(np.int64)
    sel = np.all(np.floor(world / root).astype(np.int64) == key, axis=1)
    P, ids = world[sel], all_pose[sel]
    if len(P) < 40:
        raise SystemExit(f'该体素只有 {len(P)} 点，选别处')

    mean = P.mean(0)
    centered = P - mean
    evals, evecs = np.linalg.eigh(centered.T @ centered / len(P))
    normal = evecs[:, 0]
    offsets = centered @ normal
    split = _two_means_1d(offsets)
    if split is None:
        raise SystemExit('该体素不呈双峰')
    c_low, c_high, low_mask = split

    np.savez(out, points=P, pose_ids=ids, offsets=offsets, low_mask=low_mask,
             mean=mean, normal=normal, center_low=c_low, center_high=c_high,
             separation=c_high - c_low, voxel_key=key, root=root)
    lo = np.unique(ids[low_mask]); hi = np.unique(ids[~low_mask])
    print(f'  体素 {key}  {len(P)} 点  间距 {(c_high-c_low)*100:.1f} cm')
    print(f'  层厚 σ  {offsets[low_mask].std()*100:.1f} / {offsets[~low_mask].std()*100:.1f} cm')
    print(f'  低层关键帧 {lo.min()}-{lo.max()} ({len(lo)} 帧)   '
          f'高层 {hi.min()}-{hi.max()} ({len(hi)} 帧)')
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
