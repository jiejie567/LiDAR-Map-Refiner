"""Position ATE vs M2DGR Leica GT (position-only, per-session clock offset)."""
from __future__ import annotations

import sys

import numpy as np

CLOCK_OFFSET = {'hall_04': 1.65, 'hall_02': 0.10}


def load_tum_xyz(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]


def quat_to_rot(q):
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def ate(est_tum: str, gt_txt: str, seq: str) -> tuple[float, int]:
    data = np.loadtxt(est_tum)
    ts_e, p_e, q_e = data[:, 0], data[:, 1:4], data[:, 4:8]
    ts_g, p_g = load_tum_xyz(gt_txt)
    ts_g = ts_g + CLOCK_OFFSET.get(seq, 0.0)
    idx = np.searchsorted(ts_g, ts_e)
    idx = np.clip(idx, 1, len(ts_g) - 1)
    left, right = ts_g[idx - 1], ts_g[idx]
    use_left = (ts_e - left) < (right - ts_e)
    nearest = np.where(use_left, idx - 1, idx)
    dt = np.abs(ts_g[nearest] - ts_e)
    ok = dt < 0.25
    if ok.sum() < 10:
        raise RuntimeError(f'only {ok.sum()} associations')
    rot = quat_to_rot(q_e[ok])
    p_body, dst = p_e[ok], p_g[nearest][ok]

    # Jointly solve prism lever arm (body frame) and rigid alignment.
    lever = np.zeros(3)
    for _ in range(8):
        src = p_body + np.einsum('nij,j->ni', rot, lever)
        R, t = umeyama(src, dst)
        A = np.einsum('ij,njk->nik', R, rot).reshape(-1, 3)
        b = (dst - (p_body @ R.T + t)).reshape(-1)
        lever, *_ = np.linalg.lstsq(A, b, rcond=None)
    src = p_body + np.einsum('nij,j->ni', rot, lever)
    R, t = umeyama(src, dst)
    err = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return float(np.sqrt(np.mean(err**2))), int(ok.sum())


if __name__ == '__main__':
    rmse, n = ate(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f'ATE RMSE = {rmse*100:.1f} cm over {n} associated keyframes')
