"""Runtime (Backend) – manages GPU execution of compiled kernels.

Responsibilities:
  - Allocate input/output tensors on the target device using kernel metadata.
  - Execute the PyTorch reference to obtain the *golden output*.
  - Launch the compiled Triton kernel to obtain the *test output*.
  - Handle timeouts and device-level crashes gracefully.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import shutil
import sys
import tempfile
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
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
        if meta.get("use_dot"):
            return self._allocate_dot_inputs(kernel)
        if meta.get("use_gather"):
            return self._allocate_gather_inputs(kernel)

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

    def _allocate_dot_inputs(self, kernel: GeneratedKernel) -> list[torch.Tensor]:
        """Allocate 2D matrix inputs for a dot-product kernel."""
        meta = kernel.metadata
        M = meta["dot_M"]
        N = meta["dot_N"]
        K = meta["dot_K"]
        dt_str = meta["input_dtypes"][0]
        dt = _TORCH_DTYPE_MAP.get(dt_str, torch.float32)

        A = torch.randn(M, K, device=self._device, dtype=dt)
        B = torch.randn(K, N, device=self._device, dtype=dt)
        return [A, B]

    def _allocate_gather_inputs(self, kernel: GeneratedKernel) -> list[torch.Tensor]:
        """Allocate index + data tensors for a gather/scatter kernel.

        Input 0 is an int32 index tensor whose values are clamped to
        ``[0, n_elements)`` so all gather accesses are in-bounds.
        Remaining inputs are data tensors of the appropriate dtypes.
        """
        meta = kernel.metadata
        n_elements = meta.get("n_elements", 1024)
        num_inputs = meta.get("num_inputs", 2)
        dtype_strs: list[str] = meta.get("input_dtypes", ["torch.int32"] + ["torch.float32"] * (num_inputs - 1))

        tensors: list[torch.Tensor] = []

        # Input 0: index tensor (int32, values in [0, n_elements))
        idx = torch.randint(0, n_elements, (n_elements,), device=self._device, dtype=torch.int32)
        tensors.append(idx)

        # Inputs 1..N: data tensors
        for i in range(1, num_inputs):
            dt_str = dtype_strs[i] if i < len(dtype_strs) else "torch.float32"
            dt = _TORCH_DTYPE_MAP.get(dt_str, torch.float32)

            if dt.is_floating_point:
                t = torch.randn(n_elements, device=self._device, dtype=dt)
            else:
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
        """Launch the compiled Triton kernel and return its output."""
        meta = kernel.metadata
        if meta.get("use_dot"):
            return self._run_dot_triton_kernel(kernel, compiled, inputs)

        n_elements = meta.get("n_elements", 1024)
        block_size = meta.get("block_size", 256)
        output_dtype_str = meta.get("output_dtype", "torch.float32")
        output_dtype = _TORCH_DTYPE_MAP.get(output_dtype_str, torch.float32)

        # Initialise output tensor — atomic ops require specific init values
        # so that the first write is an identity (non-overlapping pattern).
        output_init = meta.get("output_init", "empty")
        if output_init == "zeros":
            out = torch.zeros(n_elements, device=self._device, dtype=output_dtype)
        elif output_init == "neg_inf":
            if output_dtype.is_floating_point:
                fill = float("-inf")
            else:
                fill = torch.iinfo(output_dtype).min
            out = torch.full((n_elements,), fill, device=self._device, dtype=output_dtype)
        elif output_init == "pos_inf":
            if output_dtype.is_floating_point:
                fill = float("inf")
            else:
                fill = torch.iinfo(output_dtype).max
            out = torch.full((n_elements,), fill, device=self._device, dtype=output_dtype)
        else:
            out = torch.empty(n_elements, device=self._device, dtype=output_dtype)

        # The compiled kernel needs to be launched via the JIT function.
        # Write source to a temp file and import it — Triton requires @jit
        # functions to live in real Python files.
        jit_fn = self._load_jit_fn_from_source(kernel)

        grid = ((n_elements + block_size - 1) // block_size,)
        jit_fn[grid](*inputs, out, n_elements, BLOCK_SIZE=block_size)

        return out

    def _run_dot_triton_kernel(
        self,
        kernel: GeneratedKernel,
        compiled: CompilationResult,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Launch a dot-product kernel with 2D grid and stride args."""
        meta = kernel.metadata
        M = meta["dot_M"]
        N = meta["dot_N"]
        K = meta["dot_K"]
        BLOCK_M = meta["dot_BLOCK_M"]
        BLOCK_N = meta["dot_BLOCK_N"]
        BLOCK_K = meta["dot_BLOCK_K"]

        output_dtype_str = meta.get("output_dtype", "torch.float32")
        output_dtype = _TORCH_DTYPE_MAP.get(output_dtype_str, torch.float32)

        A, B = inputs[0], inputs[1]
        out = torch.empty(M, N, device=self._device, dtype=output_dtype)

        jit_fn = self._load_jit_fn_from_source(kernel)

        grid = (M // BLOCK_M, N // BLOCK_N)
        jit_fn[grid](
            A, B, out,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return out

    # --------------------------------------------------------------------- #
    # Module loading helpers                                                 #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _import_module_from_path(module_name: str, path: Path) -> types.ModuleType:
        """Dynamically import a Python module from an absolute file path."""
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _load_jit_fn_from_source(self, kernel: GeneratedKernel) -> Any:
        """Write kernel source to a temp file, import it, and return the JIT function."""
        meta = kernel.metadata
        fn_name = meta.get("kernel_fn_name", f"triton_kernel_seed_{kernel.seed}")
        # Use a unique module name to avoid collisions across seeds/reductions.
        module_name = f"_tritonfuzz_rt_{fn_name}_{id(kernel)}"
        tmp_dir = Path(tempfile.mkdtemp(prefix="tritonfuzz_rt_"))
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        src_path = tmp_dir / f"{module_name}.py"
        src_path.write_text(kernel.triton_source)

        mod = self._import_module_from_path(module_name, src_path)
        jit_fn = getattr(mod, fn_name, None)
        if jit_fn is None:
            raise RuntimeError(
                f"Triton kernel function '{fn_name}' not found in imported module"
            )
        return jit_fn
