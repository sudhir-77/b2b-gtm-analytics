"""Reproduce every result: python run.py

Use python run.py --quick for the same pipeline on a smaller simulated panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hardware import limit_threads
from src.pipeline import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulated B2B SaaS growth analytics")
    parser.add_argument("--quick", action="store_true", help="smaller panel for tests and smoke runs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="skip the pytest coverage pass (always skipped for --quick)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    limit_threads(4)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("cuda was requested but is not available")
    run(
        quick=args.quick,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        model_dir=args.model_dir,
        config_path=args.config,
        measure_coverage=not args.skip_coverage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
