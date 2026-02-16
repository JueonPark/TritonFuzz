"""Main fuzzing loop – orchestrates Generator → Driver → Runtime → Oracle."""

from __future__ import annotations

import logging
from typing import Optional

from tritonfuzz.config import FuzzConfig
from tritonfuzz.driver import CompilerDriver, CompilationResult, SweepResult
from tritonfuzz.generator import Generator, GeneratedKernel
from tritonfuzz.oracle import CampaignStats, Oracle, Verdict, VerificationResult
from tritonfuzz.reducer import Reducer
from tritonfuzz.runtime import ExecutionResult, Runtime
from tritonfuzz.target import TargetContext

logger = logging.getLogger(__name__)


class Fuzzer:
    """Top-level controller that drives one fuzzing campaign.

    Usage::

        cfg = FuzzConfig(seed_start=0, seed_end=500)
        fuzzer = Fuzzer(cfg)
        stats = fuzzer.run()
        print(stats.summary())
    """

    def __init__(self, config: FuzzConfig) -> None:
        self._config = config

        # Resolve target once and share across components
        self._target_ctx = TargetContext(target_spec=config.target_spec)

        # Instantiate the four primary modules
        self._generator = Generator()
        self._driver = CompilerDriver(config, self._target_ctx)
        self._runtime = Runtime(config)
        self._oracle = Oracle(config)
        self._reducer = Reducer(
            config,
            driver=self._driver,
            runtime=self._runtime,
            oracle=self._oracle,
        )

        self._stats = CampaignStats()

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def run(self) -> CampaignStats:
        """Execute the full campaign and return aggregate statistics."""
        seeds = self._resolve_seeds()
        logger.info(
            "Starting TritonFuzz campaign – %d seeds, target=%s",
            len(seeds),
            self._target_ctx.target,
        )

        for seed in seeds:
            result = self._fuzz_one(seed)
            self._stats.record(result)

            if result.verdict in (Verdict.FAIL, Verdict.CRASH):
                logger.warning("seed %d → %s: %s", seed, result.verdict.value, result.message)
            else:
                logger.debug("seed %d → %s", seed, result.verdict.value)

        logger.info("Campaign finished. %s", self._stats.summary())
        return self._stats

    # --------------------------------------------------------------------- #
    # Single-iteration pipeline                                               #
    # --------------------------------------------------------------------- #

    def _fuzz_one(self, seed: int) -> VerificationResult:
        """Run a single seed through the full pipeline.

        Steps:
          1. Generate kernel + reference  (Generator)
          2. Compile the Triton kernel     (Driver)
          3. Execute both on GPU           (Runtime)
          4. Verify outputs                (Oracle)
          5. Reduce on failure             (Reducer – optional)

        When ``config.sweep_count > 0`` the Driver compiles many variants
        and we cross-check for divergence (§5.5).
        """
        # Step 1 – AST Synthesis
        kernel: GeneratedKernel = self._generator.generate(seed)

        # Step 2 – Compilation (single-shot or sweep)
        if self._config.sweep_count > 0:
            return self._fuzz_one_sweep(seed, kernel)

        compiled: CompilationResult = self._driver.compile(kernel)
        if not compiled.success:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.COMPILE_ERROR,
                message=str(compiled.error),
            )

        # Step 3 – Execution
        exec_result: ExecutionResult = self._runtime.execute(kernel, compiled)

        # Step 4 – Verification
        verdict: VerificationResult = self._oracle.verify(exec_result)

        # Step 5 – Reduction (optional)
        if (
            verdict.verdict in (Verdict.FAIL, Verdict.CRASH, Verdict.NAN_MISMATCH)
            and self._config.reduce_on_failure
        ):
            reduction = self._reducer.reduce(
                kernel, compiled, exec_result, verdict=verdict,
            )
            if reduction is not None:
                logger.info(
                    "seed %d reduced: %d → %d lines%s",
                    seed,
                    reduction.original_line_count,
                    reduction.reduced_line_count,
                    " (precision-specific)" if reduction.precision_specific else "",
                )

        return verdict

    def _fuzz_one_sweep(self, seed: int, kernel: GeneratedKernel) -> VerificationResult:
        """Compile one kernel under many configs and check for divergence (§5.5).

        A *divergence bug* is reported when Variant A produces output
        that disagrees with Variant B beyond tolerance.
        """
        sweep: SweepResult = self._driver.compile_sweep(kernel)

        if not sweep.any_succeeded:
            return VerificationResult(
                seed=seed,
                verdict=Verdict.COMPILE_ERROR,
                message="All sweep variants failed to compile",
            )

        # Execute every successfully-compiled variant
        exec_results: list[tuple[CompilationResult, ExecutionResult]] = []
        for variant in sweep.variants:
            if not variant.success:
                continue
            er = self._runtime.execute(kernel, variant)
            exec_results.append((variant, er))

        if not exec_results:
            return VerificationResult(seed=seed, verdict=Verdict.CRASH, message="No variant executed")

        # Verify each variant against the golden (torch ref) output
        first_verdict = self._oracle.verify(exec_results[0][1])

        # Cross-check: compare every pair of successful variant outputs
        for i in range(1, len(exec_results)):
            vi = exec_results[i][1]
            v0 = exec_results[0][1]
            if (
                v0.test_output is not None
                and vi.test_output is not None
                and not self._oracle.tensors_close(v0.test_output, vi.test_output)
            ):
                opts_0 = exec_results[0][0].compile_options
                opts_i = exec_results[i][0].compile_options
                return VerificationResult(
                    seed=seed,
                    verdict=Verdict.FAIL,
                    message=(
                        f"Divergence bug: variant 0 ({opts_0}) vs "
                        f"variant {i} ({opts_i}) disagree"
                    ),
                )

        return first_verdict

    # --------------------------------------------------------------------- #
    # Helpers                                                                 #
    # --------------------------------------------------------------------- #

    def _resolve_seeds(self) -> list[int]:
        """Return the list of seeds to iterate over."""
        if self._config.seed_list is not None:
            return self._config.seed_list
        return list(range(self._config.seed_start, self._config.seed_end))
