"""Project-level sitecustomize to expose user strategies in subprocesses."""

from __future__ import annotations

import sys
from pathlib import Path


def _extend_path() -> None:
    strategies_dir = Path(__file__).resolve().parent / "user_data" / "strategies"
    if strategies_dir.is_dir():
        path_str = str(strategies_dir)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_extend_path()
