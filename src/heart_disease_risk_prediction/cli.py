from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible modeling and dashboard outputs for the heart disease project."
    )
    parser.add_argument("input_path", help="Path to the heart disease CSV file.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data",
        help="Directory where analytical outputs will be written.",
    )
    args = parser.parse_args()
    run_pipeline(Path(args.input_path), Path(args.output_dir))
