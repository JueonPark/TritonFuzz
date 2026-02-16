"""Runtime (Backend) – manages GPU execution of compiled kernels.

Responsibilities:
  - Allocate input/output tensors on the target device.
  - Execute the PyTorch reference to obtain the *golden output*.
  - Launch the compiled Triton kernel to obtain the *test output*.
  - Handle timeouts and device-level crashes gracefully.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from tritonfuzz.config import FuzzConfig
from tritonfuzz.driver import CompilationResult
from tritonfuzz.generator import GeneratedKernel


@dataclass
class ExecutionResult:
    """Holds the outputs (or errors) produced by a single kernel launch."""

    seed: int
    golden_output: Optional[torch.Tensor] = None
    test_output: Optional[torch.Tensor] = None
    error: Optional[BaseException] = None
    timed_out: bool = False
    metadata: dict = field(default_factory=dict)


class Runtime:
    """Backend that allocates tensors and launches kernels on the device."""

    def __init__(self, config: FuzzConfig, device: str = "cuda") -> None:
        self._config = config
        self._device = device

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def execute(
        self,
        kernel: GeneratedKernel,
        compiled: CompilationResult,
    ) -> ExecutionResult:
        """Run both reference and Triton kernel, returning their outputs.

        Parameters
        ----------
        kernel:
            The generated test-case (carries source + metadata).
        compiled:
            The compilation result (carries the binary handle).

        Returns
        -------
        ExecutionResult
        """
        try:
            inputs = self._allocate_inputs(kernel)
            golden = self._run_torch_ref(kernel, inputs)
            test = self._run_triton_kernel(compiled, inputs)
            return ExecutionResult(
                seed=kernel.seed,
                golden_output=golden,
                test_output=test,
                metadata=dict(kernel.metadata),
            )
        except TimeoutError:
            return ExecutionResult(seed=kernel.seed, timed_out=True)
        except Exception as exc:
            return ExecutionResult(seed=kernel.seed, error=exc)

    # --------------------------------------------------------------------- #
    # Private helpers                                                        #
    # --------------------------------------------------------------------- #

    def _allocate_inputs(self, kernel: GeneratedKernel) -> dict[str, torch.Tensor]:
        """Create input tensors on ``self._device``.

        TODO: Derive shapes / dtypes from kernel metadata.
        """
        # Placeholder: two random float32 vectors
        n = 1024
        return {
            "x": torch.randn(n, device=self._device),
            "y": torch.randn(n, device=self._device),
        }

    def _run_torch_ref(
        self,
        kernel: GeneratedKernel,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Execute the PyTorch reference and return the golden output.

        TODO: Dynamically exec the reference source and call the function.
        """
        raise NotImplementedError("Dynamic torch-ref execution pending")

    def _run_triton_kernel(
        self,
        compiled: CompilationResult,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Launch the compiled Triton kernel and return its output.

        TODO: Wire up real kernel launch with grid and arguments.
        """
        raise NotImplementedError("Dynamic Triton kernel launch pending")
