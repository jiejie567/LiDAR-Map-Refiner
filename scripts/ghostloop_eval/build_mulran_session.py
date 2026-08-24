"""Build a tool-format session from MulRan Sejong (HF mirror, subset download).

Steps: pick keyframes every STEP_M along GT, download those Ouster bins,
auto-validate the base->ouster extrinsic (yaw-180 vs identity, by map
planarity on a probe window), write key_point_frame/ + TUM(GT) + params.
Drift injection is done separately by make_drifted_session.py.
"""
from __future__ import annotations

import concurrent.futures as cf
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

GUI_DIR = Path('/home/anyverse/OneDrive/icra2027/slam/tools/manual_loop_closure/gui')
sys.path.insert(0, str(GUI_DIR))
from scipy.spatial.transform import Rotation  # noqa: E402
from merge_pcds import PCDCloud, write_pcd  # noqa: E402

SEQ = sys.argv[1] if len(sys.argv) > 1 else 'Sejong01'
STEP_M = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
DATA = Path('/home/anyverse/icra2027_runtime/datasets/mulran_sejong01')
OUT = Path(f'/home/anyverse/icra2027_runtime/experiments/ghostloop_eval/mulran_{SEQ.lower()}_gt')
BASE_URL = f'https://huggingface.co/datasets/WWZzz/mulran_sejong/resolve/main/{SEQ}/Ouster'
VOXEL = 0.25


def log(msg):
    print(msg, flush=True)


g = np.loadtxt(DATA / 'global_pose.csv', delimiter=',')
ts_gt = g[:, 0] * 1e-9
R_flat = g[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
P_gt = g[:, [4, 8, 12]]
P_gt = P_gt - P_gt[0]  # local origin to keep coordinates small

stamps_ns = np.loadtxt(DATA / 'ouster_front_stamp.csv', dtype=np.int64, usecols=0)
ts_scan = stamps_ns * 1e-9

# keyframe selection along GT arc length, mapped to nearest scan
sel = []
acc = 0.0
last = P_gt[0, :2]
scan_idx = np.searchsorted(ts_gt, ts_scan)
scan_idx = np.clip(scan_idx, 0, len(ts_gt) - 1)
valid = (ts_scan >= ts_gt[0]) & (ts_scan <= ts_gt[-1])
prev_pos = None
for k in np.where(valid)[0]:
    p = P_gt[scan_idx[k], :2]
    if prev_pos is None or np.linalg.norm(p - prev_pos) >= STEP_M:
        sel.append(k)
        prev_pos = p
log(f'{SEQ}: {len(sel)} keyframes at {STEP_M} m spacing')

# GT pose (base frame) interpolated at scan stamps (nearest; 100 Hz GT)
def gt_pose(k):
    i = scan_idx[k]
    Rm = R_flat[i].reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = P_gt[i]
    return T

# download bins
bin_dir = DATA / 'bins'
bin_dir.mkdir(exist_ok=True)

def fetch(k):
    name = f'{stamps_ns[k]}.bin'
    dst = bin_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return True
    r = subprocess.run(['curl', '-sL', '--fail', '-o', str(dst), f'{BASE_URL}/{name}'],
                       capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0

log('downloading bins ...')
ok = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for done, good in enumerate(ex.map(fetch, sel), 1):
        ok += bool(good)
        if done % 500 == 0:
            log(f'  {done}/{len(sel)} (ok {ok})')
log(f'downloaded {ok}/{len(sel)}')
sel = [k for k in sel if (bin_dir / f'{stamps_ns[k]}.bin').exists()]

def load_bin(k):
    raw = np.fromfile(bin_dir / f'{stamps_ns[k]}.bin', dtype=np.float32)
    return raw.reshape(-1, 4)[:, :3].astype(np.float64)


def voxel_down(pts, leaf):
    keys = np.floor(pts / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]

# extrinsic auto-validation on a probe window: compose 30 consecutive scans
# under candidate T_base_ouster, pick lowest median local roughness
candidates = {
    'identity': np.eye(4),
    'yaw180_lift': np.array([
        [-1.0, 0.0, 0.0, 1.7042],
        [0.0, -1.0, 0.0, -0.021],
        [0.0, 0.0, 1.0, 1.8047],
        [0.0, 0.0, 0.0, 1.0],
    ]),
}
probe = sel[100:130]
from scipy.spatial import cKDTree  # noqa: E402
best_name, best_score = None, None
for name, T_bo in candidates.items():
    parts = []
    for k in probe:
        pts = load_bin(k)
        pts = pts[np.linalg.norm(pts, axis=1) < 60]
        pts = voxel_down(pts, 0.5)
        Tw = gt_pose(k) @ T_bo
        parts.append(pts @ Tw[:3, :3].T + Tw[:3, 3])
    m = np.vstack(parts)
    m = m[::3]
    tree = cKDTree(m)
    idx = np.random.default_rng(0).choice(len(m), 4000, replace=False)
    _, nn = tree.query(m[idx], k=12, workers=-1)
    nb = m[nn]
    cc = nb - nb.mean(axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum('nki,nkj->nij', cc, cc) / nb.shape[1])[:, 0]
    score = float(np.sqrt(np.median(ev)))
    log(f'extrinsic {name}: probe roughness {score*100:.2f} cm')
    if best_score is None or score < best_score:
        best_name, best_score = name, score
T_bo = candidates[best_name]
log(f'chosen extrinsic: {best_name}')

# write session
OUT.mkdir(parents=True, exist_ok=True)
kf_dir = OUT / 'key_point_frame'
kf_dir.mkdir(exist_ok=True)
tum_lines = []
g2o_v, g2o_e = [], []
INFO = '1000 0 0 0 0 0 1000 0 0 0 0 100 0 0 0 4000 0 0 4000 0 4000'
prev_T = None
for i, k in enumerate(sel):
    pts = load_bin(k)
    rng_mask = np.linalg.norm(pts, axis=1) < 100.0
    pts = voxel_down(pts[rng_mask], VOXEL)
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
    arr = np.empty(pts.shape[0], dtype=dt)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    cloud = PCDCloud(
        fields=('x', 'y', 'z'), sizes=(4, 4, 4), types=('F', 'F', 'F'),
        counts=(1, 1, 1), data_type='binary', version='0.7',
        viewpoint=(0, 0, 0, 1, 0, 0, 0), comments=(), data=arr,
    )
    write_pcd(kf_dir / f'{i}.pcd', cloud)
    T = gt_pose(k) @ T_bo  # world pose of the OUSTER frame (scans are local)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    t = T[:3, 3]
    tum_lines.append(f'{ts_scan[k]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} '
                     f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}')
    g2o_v.append(f'VERTEX_SE3:QUAT {i} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} '
                 f'{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f}')
    if prev_T is not None:
        rel = np.linalg.inv(prev_T) @ T
        rq = Rotation.from_matrix(rel[:3, :3]).as_quat()
        rt = rel[:3, 3]
        g2o_e.append(f'EDGE_SE3:QUAT {i-1} {i} {rt[0]:.12f} {rt[1]:.12f} {rt[2]:.12f} '
                     f'{rq[0]:.12f} {rq[1]:.12f} {rq[2]:.12f} {rq[3]:.12f} {INFO}')
    prev_T = T
    if (i + 1) % 500 == 0:
        log(f'  wrote {i+1}/{len(sel)} keyframes')

(OUT / 'optimized_poses_tum.txt').write_text('\n'.join(tum_lines) + '\n')
(OUT / 'pose_graph.g2o').write_text('\n'.join(g2o_v + g2o_e) + '\n')
(OUT / 'runtime_params.yaml').write_text(
    'scan_context:\n'
    '  num_rings: 20\n'
    '  num_sectors: 60\n'
    '  max_radius: 80\n'
    '  dual_z_layer_enable: false\n'
    '  gravity_canonicalization_enable: false\n'
)
log(f'session ready: {OUT} ({len(sel)} kf)')
