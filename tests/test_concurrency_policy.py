from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure.concurrency import hypothesis_workers, nn_query_workers


def test_default_hypothesis_workers_leave_capacity_and_cap_at_eight() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert hypothesis_workers(cpu_count=14) == 8
        assert hypothesis_workers(cpu_count=6) == 4
        assert hypothesis_workers(cpu_count=2) == 1
        assert hypothesis_workers(cpu_count=1) == 1


def test_benchmark_overrides_are_explicit_positive_integers() -> None:
    with patch.dict(os.environ, {
            "GHOSTLOOP_NN_WORKERS": "3",
            "GHOSTLOOP_HYPOTHESIS_WORKERS": "4",
    }, clear=True):
        assert nn_query_workers() == 3
        assert hypothesis_workers(cpu_count=14) == 4


def test_invalid_overrides_fall_back_to_safe_defaults() -> None:
    for invalid in ("", "0", "-2", "not-an-integer"):
        with patch.dict(os.environ, {
                "GHOSTLOOP_NN_WORKERS": invalid,
                "GHOSTLOOP_HYPOTHESIS_WORKERS": invalid,
        }, clear=True):
            assert nn_query_workers() == 1
            assert hypothesis_workers(cpu_count=6) == 4
