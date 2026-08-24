"""Dependency-free concurrency policy for registration.

Kept next to ``repair_presets`` and for the same reason: the GUI, the headless
evaluator and any benchmark must not drift apart on how many threads they use.
This module imports no Open3D, SciPy, Qt or NumPy code.

Two knobs, and they pull against each other:

``nn_query_workers``
    The ``workers=`` argument SciPy's ``cKDTree.query`` receives. A gated ICP
    issues one query per iteration -- thirty for the near tier, a hundred for
    the wide one -- and each call sets up and tears down its own thread team.
    Measured on four sequences spanning 16-ring simulation scans to 289k-point
    outdoor targets, that per-call overhead outweighs the parallel search at
    every worker count above one, so a single-threaded query is fastest
    everywhere. The previous value, ``cpu_count - 4``, was a pessimisation
    worth 1.26-1.39x.

``hypothesis_workers``
    How many of a pair's initial guesses register concurrently. They are
    independent -- same source, same target, different starting transform.
    Their cKDTree queries do not create nested teams, although Open3D voxel
    preparation and linear-algebra kernels can still use internal threads; the
    bounded worker count limits that pressure rather than claiming exclusive
    cores.

Throughput per registration, four sequences, 14 cores, against the previous
(nn=10, serial hypotheses) setting:

    simulation 1.66x   in-house indoor 2.00x   in-house outdoor 1.72x
    MCD kth_night_01 2.18x

Both values are overridable through the environment for benchmarking:
``GHOSTLOOP_NN_WORKERS`` and ``GHOSTLOOP_HYPOTHESIS_WORKERS``.
"""

from __future__ import annotations

import os


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def nn_query_workers() -> int:
    """Threads for one ``cKDTree.query`` call."""
    return _env_int('GHOSTLOOP_NN_WORKERS') or 1


def hypothesis_workers(cpu_count: int | None = None) -> int:
    """Initial guesses to register concurrently for one pair.

    Keeps the Python-level task count at least two below the logical CPU count
    and stops at eight. This is a scheduling bound, not a hard core reservation:
    native kernels may use additional threads. Full-pipeline A/B retained byte-
    identical constraints and trajectories while reducing wall time 785→335 s
    on escalator00 and 348→204 s on simulation.
    """
    override = _env_int('GHOSTLOOP_HYPOTHESIS_WORKERS')
    if override is not None:
        return override
    cores = cpu_count if cpu_count and cpu_count > 0 else (os.cpu_count() or 4)
    return max(1, min(8, cores - 2))
