#!/usr/bin/env python3
"""量化"地图不一致"与"外观检索"这两个证据通道的重叠。

论文 C1 的立论是它们正交。要说清楚需要三个数：
  1. 两条流水线各自接受了多少约束、交集多大
  2. 我们接受的约束里，有多少压根不在检索的候选池内 —— 不是被相似度阈值否掉，
     而是从未被提名，因为两端互相都不是对方的最近邻
  3. 要把这些都覆盖到，检索的候选池得放开到多大

候选池取自 scloop 那次跑日志里 AutoSeed 打印的全部候选对（描述子检索给出的
排序表），接受集取自各自日志里 ACCEPT 的行。

用法：channel_overlap.py <ghost跑日志> <scloop跑日志>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ACC = re.compile(r'^\[(?:AutoSeed|Round\d+)\] (\d+)->(\d+): fit=')
CAND = re.compile(r'(\d+)->(\d+)\(d=([\d.]+),yaw=(-?\d+)\)')


def accepted(log: Path) -> set[tuple[int, int]]:
    out = set()
    for line in log.read_text(errors='ignore').splitlines():
        m = ACC.match(line)
        if m:
            out.add(tuple(sorted((int(m.group(1)), int(m.group(2))))))
    return out


def candidates(log: Path) -> dict[tuple[int, int], float]:
    out = {}
    for line in log.read_text(errors='ignore').splitlines():
        if 'candidates:' not in line:
            continue
        for a, b, d, _ in CAND.findall(line):
            key = tuple(sorted((int(a), int(b))))
            out[key] = min(out.get(key, 9.9), float(d))
    return out


def main() -> None:
    ghost_log, sc_log = Path(sys.argv[1]), Path(sys.argv[2])
    g, s = accepted(ghost_log), accepted(sc_log)
    pool = candidates(sc_log) or candidates(ghost_log)
    shared = g & s

    # near-duplicate 容差：同一处回环两边可能落在相邻关键帧上
    def close(p, q, tol=25):
        return (abs(p[0]-q[0]) <= tol and abs(p[1]-q[1]) <= tol) or \
               (abs(p[0]-q[1]) <= tol and abs(p[1]-q[0]) <= tol)
    shared_fuzzy = {p for p in g if any(close(p, q) for q in s)}
    outside = {p for p in g if not any(close(p, q) for q in pool)}

    print(f'  鬼影流水线接受   {len(g)} 条')
    print(f'  纯检索接受       {len(s)} 条')
    print(f'  严格交集         {len(shared)} 条')
    print(f'  容差 25 帧内交集 {len(shared_fuzzy)} 条')
    print(f'  检索候选池大小   {len(pool)} 对')
    print(f'  我们接受但不在候选池内（从未被提名）: {len(outside)} 条')
    if outside:
        print('    ' + ' '.join(f'{a}->{b}' for a, b in sorted(outside)[:12]))


if __name__ == '__main__':
    main()
