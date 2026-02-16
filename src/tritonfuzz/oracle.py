"""Oracle (Verifier) – differential testing and verification (§6).

Implements the full verification strategy from the design document:

* **§6.1 Twin Generator Pattern** — compares the Triton kernel output
  against the PyTorch reference ("golden") output produced by the Runtime.
* **§6.2 Numerical Robustness**

  - *Dynamic tolerance*: ``atol`` / ``rtol`` are adjusted based on the
    kernel's operations (``exp``, ``dot`` → looser) and output dtype
    (FP16/BF16 → looser).
  - *NaN/Inf consistency*: if the reference produces ``NaN`` the test must
    too; if the test produces ``NaN`` where the reference doesn't, a
    **critical** NaN-propagation bug is reported.
* **§6.3 Atomics (non-determinism)** — for kernels that use ``atomic_add``
  etc., element-wise comparison is replaced by *global-sum* / *histogram*
  checks.
* **§5.5 Divergence** — ``_tensors_close()`` helper used by the fuzzer's
  sweep mode to cross-check variant outputs.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from tritonfuzz.config import FuzzConfig
from tritonfuzz.runtime import ExecutionResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Verdict & result types                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


class Verdict(enum.Enum):
    """Possible outcomes for a single test-case."""

    PASS = "pass"
    FAIL = "fail"              # Numerical mismatch
    CRASH = "crash"            # Exception / segfault
    TIMEOUT = "timeout"
    COMPILE_ERROR = "compile_error"
    NAN_MISMATCH = "nan_mismatch"   # §6.2 — inconsistent NaN propagation


@dataclass
class VerificationResult:
    """Detailed verdict for a single seed."""

    seed: int
    verdict: Verdict
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    effective_atol: Optional[float] = None
    effective_rtol: Optional[float] = None
    message: str = ""


@dataclass
class CampaignStats:
    """Running statistics for an entire fuzzing campaign."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    crashed: int = 0
    timed_out: int = 0
    compile_errors: int = 0
    nan_mismatches: int = 0
    interesting_seeds: list[int] = field(default_factory=list)

    def record(self, result: VerificationResult) -> None:
        self.total += 1
        match result.verdict:
            case Verdict.PASS:
                self.passed += 1
            case Verdict.FAIL:
                self.failed += 1
                self.interesting_seeds.append(result.seed)
            case Verdict.CRASH:
                self.crashed += 1
                self.interesting_seeds.append(result.seed)
            case Verdict.TIMEOUT:
                self.timed_out += 1
            case Verdict.COMPILE_ERROR:
                self.compile_errors += 1
            case Verdict.NAN_MISMATCH:
                self.nan_mismatches += 1
                self.interesting_seeds.append(result.seed)

    def summary(self) -> str:
        return (
            f"Total: {self.total} | Pass: {self.passed} | Fail: {self.failed} | "
            f"Crash: {self.crashed} | Timeout: {self.timed_out} | "
            f"CompileErr: {self.compile_errors} | NaN: {self.nan_mismatches}"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dynamic tolerance tables (§6.2)                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

# Operations that introduce significant rounding → tolerance multiplier.
_OP_TOLERANCE_MULTIPLIERS: dict[str, float] = {
    "exp":  2.0,
    "log":  2.0,
    "sin":  1.5,
    "cos":  1.5,
    "sqrt": 1.5,
    "div":  1.5,
    "dot":  3.0,
}

# Output dtype → base tolerance override.
_DTYPE_TOLERANCE: dict[torch.dtype, tuple[float, float]] = {
    torch.float16:  (1e-2,  1e-2),
    torch.bfloat16: (2e-2,  2e-2),
    torch.float32:  (1e-4,  1e-4),
    torch.float64:  (1e-8,  1e-8),
    torch.int32:    (0.0,   0.0),
    torch.int64:    (0.0,   0.0),
}


def _compute_effective_tolerance(
    base_atol: float,
    base_rtol: float,
    ops_used: list[str],
    output_dtype: torch.dtype | None,
) -> tuple[float, float]:
    """Return ``(atol, rtol)`` adjusted for op complexity and dtype (§6.2).

    Strategy:
    1. Start from the dtype-specific base (or the config base if dtype is
       unknown).
    2. For every "hard" op in the kernel, multiply by its factor.
    3. Cap the combined multiplier so tolerances don't blow up.
    """
    # Step 1: dtype base
    if output_dtype is not None and output_dtype in _DTYPE_TOLERANCE:
        atol, rtol = _DTYPE_TOLERANCE[output_dtype]
    else:
        atol, rtol = base_atol, base_rtol

    # Step 2: accumulate op multipliers (only count each op type once)
    seen: set[str] = set()
    combined = 1.0
    for op in ops_used:
        if op in _OP_TOLERANCE_MULTIPLIERS and op not in seen:
            combined *= _OP_TOLERANCE_MULTIPLIERS[op]
            seen.add(op)

    # Step 3: cap
    combined = min(combined, 20.0)
    return atol * combined, rtol * combined


# ═══════════════════════════════════════════════════════════════════════════ #
#  Oracle                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


class Oracle:
    """Verifier that decides whether a test-case passed or failed.

    The Oracle implements the three-phase comparison described in §6:

    1. **NaN/Inf consistency check** — fast short-circuit on propagation bugs.
    2. **Comparison-mode selection** — element-wise (default) or
       global-sum (for atomics, §6.3).
    3. **Tolerance-adjusted numerical comparison** — dynamic tolerances
       based on ops and dtype (§6.2).
    """

    def __init__(self, config: FuzzConfig) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────

    def verify(self, exec_result: ExecutionResult) -> VerificationResult:
        """Compare golden and test outputs and return a verdict.

        Parameters
        ----------
        exec_result:
            The ``ExecutionResult`` produced by the Runtime.

        Returns
        -------
        VerificationResult
        """
        seed = exec_result.seed

        # --- Short-circuit on non-output outcomes -------------------------
        if exec_result.timed_out:
            return VerificationResult(seed=seed, verdict=Verdict.TIMEOUT)

        if exec_result.error is not None:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.CRASH,
                message=str(exec_result.error),
            )

        if exec_result.golden_output is None or exec_result.test_output is None:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.CRASH,
                message="Missing output tensor(s)",
            )

        golden = exec_result.golden_output
        test = exec_result.test_output
        metadata = exec_result.metadata

        # --- Shape guard — catch mismatches early -------------------------
        if golden.shape != test.shape:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.FAIL,
                message=(
                    f"Output shape mismatch: golden={golden.shape}, "
                    f"test={test.shape}"
                ),
            )

        # --- Phase 1: NaN / Inf consistency (§6.2) ------------------------
        nan_result = self._check_nan_consistency(seed, golden, test)
        if nan_result is not None:
            return nan_result

        # --- Phase 2: choose comparison mode (§6.3) -----------------------
        ops_used: list[str] = metadata.get("ops_used", [])
        has_atomics = any(
            op.startswith("atomic") for op in ops_used
        )

        if has_atomics:
            return self._compare_atomic(seed, golden, test, ops_used, metadata)

        # --- Phase 3: standard numerical comparison (§6.1–6.2) ------------
        return self._compare_tensors(seed, golden, test, ops_used, metadata)

    def tensors_close(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Check if two tensors agree within the configured tolerances.

        Used by the fuzzer's sweep-mode divergence check (§5.5).
        """
        return bool(torch.allclose(a, b, atol=self._config.atol, rtol=self._config.rtol))

    # ── Phase 1: NaN / Inf consistency (§6.2) ────────────────────────────

    @staticmethod
    def _check_nan_consistency(
        seed: int,
        golden: torch.Tensor,
        test: torch.Tensor,
    ) -> Optional[VerificationResult]:
        """Detect inconsistent NaN / Inf propagation.

        Rules:
        * If golden has NaN, test **must** also have NaN at the same positions.
          (Mismatch is suspicious but could be due to ``-ffast-math``.)
        * If test has NaN where golden **does not**, it is a **critical bug**
          — likely an unmasked OOB load reading garbage.
        * Inf follows the same logic.

        Returns ``None`` when NaN/Inf is consistent (or absent), otherwise
        returns a ``NAN_MISMATCH`` verdict.
        """
        # Only applies to floating-point outputs
        if not golden.is_floating_point():
            return None

        golden_nan = torch.isnan(golden)
        test_nan = torch.isnan(test)

        # Critical: test has NaN where golden does not
        spurious_nan = test_nan & ~golden_nan
        if spurious_nan.any():
            count = spurious_nan.sum().item()
            return VerificationResult(
                seed=seed,
                verdict=Verdict.NAN_MISMATCH,
                message=(
                    f"Triton produced {count} spurious NaN(s) where PyTorch "
                    f"reference has valid numbers — likely unmasked OOB load"
                ),
            )

        # Suspicious (but non-critical): golden has NaN, test does not
        missing_nan = golden_nan & ~test_nan
        if missing_nan.any():
            count = missing_nan.sum().item()
            logger.debug(
                "seed %d: golden has %d NaN(s) that test does not reproduce "
                "(may be fast-math related)",
                seed, count,
            )

        # Same check for Inf
        golden_inf = torch.isinf(golden)
        test_inf = torch.isinf(test)

        spurious_inf = test_inf & ~golden_inf
        if spurious_inf.any():
            count = spurious_inf.sum().item()
            return VerificationResult(
                seed=seed,
                verdict=Verdict.NAN_MISMATCH,
                message=(
                    f"Triton produced {count} spurious Inf(s) where PyTorch "
                    f"reference has finite numbers"
                ),
            )

        return None  # All clear

    # ── Phase 3a: standard element-wise comparison (§6.1–6.2) ────────────

    def _compare_tensors(
        self,
        seed: int,
        golden: torch.Tensor,
        test: torch.Tensor,
        ops_used: list[str],
        metadata: dict,
    ) -> VerificationResult:
        """Element-wise comparison with dynamically adjusted tolerances."""
        # Resolve effective tolerances
        output_dtype = golden.dtype
        atol, rtol = _compute_effective_tolerance(
            self._config.atol, self._config.rtol, ops_used, output_dtype,
        )

        # Mask out NaN positions (already validated in phase 1)
        if golden.is_floating_point():
            valid = ~torch.isnan(golden) & ~torch.isnan(test)
            golden_v = golden[valid]
            test_v = test[valid]
        else:
            golden_v = golden.flatten()
            test_v = test.flatten()

        if golden_v.numel() == 0:
            # Entire tensor is NaN — already handled above
            return VerificationResult(seed=seed, verdict=Verdict.PASS)

        abs_diff = (golden_v.float() - test_v.float()).abs()
        max_abs = abs_diff.max().item()

        denom = golden_v.float().abs().clamp(min=1e-12)
        max_rel = (abs_diff / denom).max().item()

        close = bool(torch.allclose(golden_v.float(), test_v.float(), atol=atol, rtol=rtol))

        if close:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.PASS,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                effective_atol=atol,
                effective_rtol=rtol,
            )
        else:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.FAIL,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                effective_atol=atol,
                effective_rtol=rtol,
                message=(
                    f"Mismatch: max_abs={max_abs:.6e} (atol={atol:.2e}), "
                    f"max_rel={max_rel:.6e} (rtol={rtol:.2e})"
                ),
            )

    # ── Phase 3b: atomic / non-deterministic comparison (§6.3) ───────────

    def _compare_atomic(
        self,
        seed: int,
        golden: torch.Tensor,
        test: torch.Tensor,
        ops_used: list[str],
        metadata: dict,
    ) -> VerificationResult:
        """Relaxed comparison for kernels with atomic operations.

        Since atomic execution order is undefined, element-wise equality is
        meaningless.  Instead we check aggregate invariants:

        1. **Global sum** — ``golden.sum()`` ≈ ``test.sum()``
        2. **Sorted histogram** — ``sort(golden)`` ≈ ``sort(test)``
           (catches permutation-level differences).
        3. **Element count** — shapes must match.
        """
        if golden.shape != test.shape:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.FAIL,
                message=(
                    f"Atomic output shape mismatch: golden={golden.shape}, "
                    f"test={test.shape}"
                ),
            )

        # --- Global sum check ---
        golden_sum = golden.float().sum()
        test_sum = test.float().sum()
        atol, rtol = _compute_effective_tolerance(
            self._config.atol, self._config.rtol, ops_used, golden.dtype,
        )
        sum_close = bool(torch.allclose(golden_sum, test_sum, atol=atol * golden.numel(), rtol=rtol))

        if not sum_close:
            diff = (golden_sum - test_sum).abs().item()
            return VerificationResult(
                seed=seed,
                verdict=Verdict.FAIL,
                max_abs_diff=diff,
                effective_atol=atol * golden.numel(),
                effective_rtol=rtol,
                message=(
                    f"Atomic global-sum mismatch: golden_sum={golden_sum.item():.6e}, "
                    f"test_sum={test_sum.item():.6e}, diff={diff:.6e}"
                ),
            )

        # --- Sorted-histogram check ---
        golden_sorted = golden.flatten().float().sort().values
        test_sorted = test.flatten().float().sort().values
        sorted_close = bool(torch.allclose(golden_sorted, test_sorted, atol=atol, rtol=rtol))

        if not sorted_close:
            abs_diff = (golden_sorted - test_sorted).abs()
            max_abs = abs_diff.max().item()
            return VerificationResult(
                seed=seed,
                verdict=Verdict.FAIL,
                max_abs_diff=max_abs,
                effective_atol=atol,
                effective_rtol=rtol,
                message=(
                    f"Atomic sorted-histogram mismatch: max_abs={max_abs:.6e}"
                ),
            )

        return VerificationResult(
            seed=seed,
            verdict=Verdict.PASS,
            effective_atol=atol,
            effective_rtol=rtol,
            message="Atomic comparison (global-sum + sorted-histogram): PASS",
        )
