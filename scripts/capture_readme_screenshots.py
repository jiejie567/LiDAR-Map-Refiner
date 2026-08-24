#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from manual_loop_closure_tool import ManualLoopClosureWindow, SelectedEdgeRef  # noqa: E402


def process_events(app: QtWidgets.QApplication, steps: int = 40, interval_ms: int = 40) -> None:
    for _ in range(steps):
        app.processEvents(QtCore.QEventLoop.AllEvents, interval_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture README screenshots from a real session.")
    parser.add_argument("--session-root", type=Path, required=True, help="Mapping session root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "screenshots",
        help="Output directory for generated screenshots",
    )
    parser.add_argument("--width", type=int, default=1600, help="Window width")
    parser.add_argument("--height", type=int, default=980, help="Window height")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication([])
    window = ManualLoopClosureWindow(initial_session_root=args.session_root)
    window.resize(args.width, args.height)
    window.show()
    process_events(app, 10)

    window.load_session()
    process_events(app, 50)
    loaded_path = args.output_dir / "session-loaded.png"
    window.grab().save(str(loaded_path))

    if window.pose_graph and window.pose_graph.loop_edges:
        edge = window.pose_graph.loop_edges[0]
        window._select_edge(SelectedEdgeRef("existing", edge.edge_uid), reset_camera=True)
        process_events(app, 80)
        edge_path = args.output_dir / "edge-selected.png"
        window.grab().save(str(edge_path))
        print(f"saved {edge_path}")

    print(f"saved {loaded_path}")
    window.close()
    process_events(app, 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
