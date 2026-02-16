"""Runtime (Backend) – manages GPU execution of compiled kernels.

Responsibilities:
  - Allocate input/output tensors on the target device using kernel metadata.
  - Execute the PyTorch reference to obtain the *golden output*.
  - Launch the compiled Triton kernel to obtain the *test output*.
  - Handle timeouts and device-level crashes gracefully.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from tritonfuzz.config import FuzzConfig
from tritonfuzz.driver import CompilationResult
from tritonfuzz.generator import GeneratedKernel

logger = logging.getLogger(__name__)

# ── Mapping from PyTorch dtype strings to actual torch.dtype objects ──────

_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.float16":  torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32":  torch.float32,
    "torch.float64":  torch.float64,
    "torch.int8":     torch.int8,
    "torch.int16":    torch.int16,
    "torch.int32":    torch.int32,
    "torch.int64":    torch.int64,
    "torch.bool":     torch.bool,
}


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
        result_holder: list[ExecutionResult] = []
        exc_holder: list[BaseException] = []

        def _inner() -> None:
            try:
                inputs = self._allocate_inputs(kernel)
                golden = self._run_torch_ref(kernel, inputs)
                test = self._run_triton_kernel(kernel, compiled, inputs)
                result_holder.append(ExecutionResult(
                    seed=kernel.seed,
                    golden_output=golden,
                    test_output=test,
                    metadata=dict(kernel.metadata),
                ))
            except Exception as exc:
                exc_holder.append(exc)

        timeout = self._config.timeout_seconds
        thread = threading.Thread(target=_inner, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning("seed %d: execution timed out after %.1fs", kernel.seed, timeout)
            return ExecutionResult(
                seed=kernel.seed,
                timed_out=True,
                metadata=dict(kernel.metadata),
            )

        if exc_holder:
            return ExecutionResult(
                seed=kernel.seed,
                error=exc_holder[0],
                metadata=dict(kernel.metadata),
            )

        if result_holder:
            return result_holder[0]

        return ExecutionResult(
            seed=kernel.seed,
            error=RuntimeError("Execution thread finished with no result"),
            metadata=dict(kernel.metadata),
        )

    # --------------------------------------------------------------------- #
    # Private helpers                                                        #
    # --------------------------------------------------------------------- #

    def _allocate_inputs(self, kernel: GeneratedKernel) -> list[torch.Tensor]:
        """Create input tensors on ``self._device`` using kernel metadata.

        Derives ``n_elements`` and per-input dtypes from the metadata
        produced by the Generator.
        """
        meta = kernel.metadata
        n_elements = meta.get("n_elements", 1024)
        num_inputs = meta.get("num_inputs", 1)
        dtype_strs: list[str] = meta.get("input_dtypes", ["torch.float32"] * num_inputs)

        tensors: list[torch.Tensor] = []
        for i in range(num_inputs):
            dt_str = dtype_strs[i] if i < len(dtype_strs) else "torch.float32"
            dt = _TORCH_DTYPE_MAP.get(dt_str, torch.float32)

            if dt.is_floating_point:
                t = torch.randn(n_elements, device=self._device, dtype=dt)
            else:
                # For integer types, use small values to avoid overflow
                info = torch.iinfo(dt)
                low = max(info.min, -127)
                high = min(info.max, 127) + 1
                t = torch.randint(low, high, (n_elements,), device=self._device, dtype=dt)

            tensors.append(t)

        return tensors

    def _run_torch_ref(
        self,
        kernel: GeneratedKernel,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Execute the PyTorch reference and return the golden output.

        Dynamically ``exec``s the reference source, extracts the function
        by name, and calls it with the allocated input tensors.
        """
        ref_ns: dict[str, Any] = {}
        exec(compile(kernel.torch_ref_source, f"<torch_ref_seed_{kernel.seed}>", "exec"), ref_ns)

        ref_fn_name = kernel.metadata.get("ref_fn_name", f"torch_ref_seed_{kernel.seed}")
        ref_fn = ref_ns.get(ref_fn_name)
        if ref_fn is None:
            raise RuntimeError(
                f"Reference function '{ref_fn_name}' not found in exec'd module"
            )

        return ref_fn(*inputs)

    def _run_triton_kernel(
        self,
        kernel: GeneratedKernel,
        compiled: CompilationResult,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Launch the compiled Triton kernel and return its output.

        Dynamically imports and calls the ``@triton.jit`` kernel using the
        standard Triton grid-launch syntax.
        """
        meta = kernel.metadata
        n_elements = meta.get("n_elements", 1024)
        block_size = meta.get("block_size", 256)
        output_dtype_str = meta.get("output_dtype", "torch.float32")
        output_dtype = _TORCH_DTYPE_MAP.get(output_dtype_str, torch.float32)

        out = torch.empty(n_elements, device=self._device, dtype=output_dtype)

        # The compiled kernel needs to be launched via the JIT function.
        # Re-exec the kernel source to get the JIT function and launch it.
        kernel_ns: dict[str, Any] = {}
        exec(
            compile(kernel.triton_source, f"<triton_kernel_seed_{kernel.seed}>", "exec"),
            kernel_ns,
        )

        fn_name = meta.get("kernel_fn_name", f"triton_kernel_seed_{kernel.seed}")
        jit_fn = kernel_ns.get(fn_name)
        if jit_fn is None:
            raise RuntimeError(
                f"Triton kernel function '{fn_name}' not found in exec'd module"
            )

        grid = ((n_elements + block_size - 1) // block_size,)
        jit_fn[grid](*inputs, out, n_elements, BLOCK_SIZE=block_size)

        return out
