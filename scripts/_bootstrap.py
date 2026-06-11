"""Helpers for running package entry points directly from the repository."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
