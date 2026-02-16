"""CLI entry-point for the ``tritonfuzz`` command."""

from __future__ import annotations

import argparse
import logging
import sys

from tritonfuzz.config import FuzzConfig
from tritonfuzz.fuzzer import Fuzzer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tritonfuzz",
        description="Fuzz the Triton compiler by generating and verifying GPU kernels.",
    )

    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed in the range (default: 0).",
    )
    p.add_argument(
        "--seed-end",
        type=int,
        default=100,
        help="One-past-the-last seed in the range (default: 100).",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Target specification: 'auto', 'cuda', 'hip', 'cpu', "
            "'cuda:90', 'hip:gfx942', etc. (default: auto-detect)."
        ),
    )
    p.add_argument(
        "--sweep-count",
        type=int,
        default=0,
        help=(
            "Compile each seed N times with different options and "
            "cross-check for divergence bugs (default: 0 = disabled)."
        ),
    )
    p.add_argument(
        "--no-randomize-options",
        action="store_true",
        help="Disable randomisation of compilation options.",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for numerical comparison (default: 1e-2).",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for numerical comparison (default: 1e-2).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="tritonfuzz_results",
        help="Directory for storing results (default: tritonfuzz_results).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Logging setup ------------------------------------------------------
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )

    # --- Build config -------------------------------------------------------
    config = FuzzConfig(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        target_spec=args.target,
        randomize_compile_options=not args.no_randomize_options,
        sweep_count=args.sweep_count,
        atol=args.atol,
        rtol=args.rtol,
        output_dir=args.output_dir,
    )

    # --- Run ----------------------------------------------------------------
    fuzzer = Fuzzer(config)
    stats = fuzzer.run()

    print(stats.summary())
    return 1 if stats.interesting_seeds else 0


if __name__ == "__main__":
    sys.exit(main())
