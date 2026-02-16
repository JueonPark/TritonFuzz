"""Global configuration for a fuzzing campaign."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class FuzzConfig:
    """Top-level knobs that control a fuzzing campaign."""

    # --- Seed control -----------------------------------------------------------
    seed_start: int = 0
    seed_end: int = 1000
    # If provided, only these specific seeds are tested.
    seed_list: Optional[list[int]] = None

    # --- Target (§5.1) ----------------------------------------------------------
    # Accepts: None/"auto", "cuda", "hip", "cpu", "cuda:90", "hip:gfx942", …
    target_spec: Optional[str] = None

    # --- Compilation options (§5.2 – §5.4) --------------------------------------
    randomize_compile_options: bool = True

    # --- Config sweeping / autotuning fuzzing (§5.5) ----------------------------
    # When > 0, each seed is compiled ``sweep_count`` times with different
    # option combinations; outputs are cross-checked for divergence.
    sweep_count: int = 0

    # --- Execution --------------------------------------------------------------
    timeout_seconds: float = 30.0

    # --- Verification -----------------------------------------------------------
    atol: float = 1e-2
    rtol: float = 1e-2

    # --- I/O --------------------------------------------------------------------
    output_dir: Path = Path("tritonfuzz_results")

    # --- Reducer ----------------------------------------------------------------
    reduce_on_failure: bool = True
    max_reduction_steps: int = 50
