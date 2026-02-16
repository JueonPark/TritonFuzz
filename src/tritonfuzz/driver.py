"""Compiler Driver (Middleware) – multi-target compilation & config sweeping.

Implements the full compilation pipeline described in §5:

* **Single-shot compilation** (§5.1–5.4):
  Generates one set of backend-specific options, writes the kernel to a temp
  module, calls ``triton.compile()``, and returns the binary handle.

* **Config sweeping** (§5.5 — autotuning fuzzing):
  Compiles the *same* kernel ``N`` times with different ``(num_warps,
  num_stages, …)`` combinations, returning all variants.  The caller
  (``Fuzzer``) can then execute each variant and cross-check for
  *divergence bugs* (output differs across tuning configs).

Backends handled:
  - **CUDA** (§5.2) — ``num_warps``, ``num_stages``, ``enable_fp_fusion``
  - **HIP**  (§5.3) — ``num_warps``, ``waves_per_eu``, ``matrix_core_version``
  - **CPU**  (§5.4) — vectorisation verification (minimal options)
"""

from __future__ import annotations

import atexit
import importlib
import logging
import random
import shutil
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tritonfuzz.config import FuzzConfig
from tritonfuzz.generator import GeneratedKernel
from tritonfuzz.target import TargetContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Data models                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #


@dataclass
class CompilationResult:
    """Outcome of a single ``triton.compile()`` invocation."""

    seed: int
    success: bool
    compiled_kernel: Any = None           # triton CompiledKernel handle
    compile_options: dict = field(default_factory=dict)
    error: Optional[BaseException] = None
    variant_index: int = 0                # index within a sweep (0 = default)


@dataclass
class SweepResult:
    """Outcome of compiling one kernel under *multiple* option variants (§5.5)."""

    seed: int
    variants: list[CompilationResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(v.success for v in self.variants)

    @property
    def any_succeeded(self) -> bool:
        return any(v.success for v in self.variants)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CompilerDriver                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #


class CompilerDriver:
    """Middleware that compiles generated Triton kernels to device binaries."""

    def __init__(self, config: FuzzConfig, target_ctx: TargetContext) -> None:
        self._config = config
        self._target_ctx = target_ctx

    # ── Public API ────────────────────────────────────────────────────────

    def compile(self, kernel: GeneratedKernel) -> CompilationResult:
        """Compile with a single (possibly randomised) option set.

        This is the default path used when ``config.sweep_count == 0``.
        """
        options = self._resolve_compile_options(kernel.seed)
        return self._try_compile(kernel, options, variant_index=0)

    def compile_sweep(self, kernel: GeneratedKernel) -> SweepResult:
        """Compile the *same* kernel under many option variants (§5.5).

        Returns a :class:`SweepResult` containing one
        :class:`CompilationResult` per variant.  Downstream, the caller
        can execute each and check for *divergence bugs*.
        """
        count = max(self._config.sweep_count, 1)
        rng = random.Random(kernel.seed)
        result = SweepResult(seed=kernel.seed)

        for idx in range(count):
            options = self._target_ctx.get_backend_options(rng)
            cr = self._try_compile(kernel, options, variant_index=idx)
            result.variants.append(cr)

        return result

    # ── Option resolution ─────────────────────────────────────────────────

    def _resolve_compile_options(self, seed: int) -> dict:
        """Build the dict of options passed to ``triton.compile()``.

        When ``config.randomize_compile_options`` is *True* the values are
        randomised deterministically, keyed on *seed*.
        """
        if not self._config.randomize_compile_options:
            return self._target_ctx.get_backend_options(rng=None)

        rng = random.Random(seed)
        return self._target_ctx.get_backend_options(rng)

    # ── Compilation ───────────────────────────────────────────────────────

    def _try_compile(
        self,
        kernel: GeneratedKernel,
        options: dict,
        *,
        variant_index: int = 0,
    ) -> CompilationResult:
        """Run ``triton.compile()`` in a try/except and return a result."""
        logger.debug(
            "Compiling seed %d (variant %d) with options %s",
            kernel.seed, variant_index, options,
        )
        try:
            compiled = self._invoke_triton_compile(kernel, options)
            return CompilationResult(
                seed=kernel.seed,
                success=True,
                compiled_kernel=compiled,
                compile_options=options,
                variant_index=variant_index,
            )
        except Exception as exc:
            logger.debug("Compilation failed for seed %d: %s", kernel.seed, exc)
            return CompilationResult(
                seed=kernel.seed,
                success=False,
                error=exc,
                compile_options=options,
                variant_index=variant_index,
            )

    def _invoke_triton_compile(self, kernel: GeneratedKernel, options: dict) -> Any:
        """Write the kernel source to a temp module, extract the JIT
        function, and call ``triton.compile()``.

        Steps:
          1. Write kernel source to a temporary ``.py`` file.
          2. Dynamically import the module to get the ``@triton.jit`` function.
          3. Call ``triton.compile(fn, signature=…, target=…, **options)``.
          4. Return the ``CompiledKernel`` handle.
        """
        import triton
        from triton.compiler.compiler import ASTSource

        # ── Step 1: persist source to a temp file ────────────────────────
        fn_name = kernel.metadata.get("kernel_fn_name", f"triton_kernel_seed_{kernel.seed}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="tritonfuzz_"))
        # Register cleanup so temp dirs don't leak over a long campaign.
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        src_path = tmp_dir / f"{fn_name}.py"
        src_path.write_text(kernel.triton_source)

        # ── Step 2: dynamically import ───────────────────────────────────
        mod = self._import_module_from_path(fn_name, src_path)
        jit_fn = getattr(mod, fn_name)

        # ── Step 3: build signature from metadata ────────────────────────
        num_inputs = kernel.metadata.get("num_inputs", 1)
        input_dtypes = kernel.metadata.get("input_dtypes")
        output_dtype = kernel.metadata.get("output_dtype")
        sig = self._build_signature(jit_fn, num_inputs, input_dtypes, output_dtype)

        # ── Step 4: build constexprs for tl.constexpr parameters ─────────
        constexprs = self._build_constexprs(kernel, jit_fn)

        # ── Step 5: wrap in ASTSource and compile ────────────────────────
        triton_target = self._target_ctx.get_triton_target()
        ast_src = ASTSource(fn=jit_fn, signature=sig, constexprs=constexprs)

        compiled = triton.compile(
            src=ast_src,
            target=triton_target,
            options=options,
        )
        return compiled

    # ── Helpers ───────────────────────────────────────────────────────────

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

    # ── Dtype → Triton signature string mapping ────────────────────────

    _TORCH_DTYPE_TO_SIG: dict[str, str] = {
        "torch.float16":  "*fp16",
        "torch.bfloat16": "*bf16",
        "torch.float32":  "*fp32",
        "torch.float64":  "*fp64",
        "torch.int8":     "*i8",
        "torch.int16":    "*i16",
        "torch.int32":    "*i32",
        "torch.int64":    "*i64",
        "torch.bool":     "*i1",
    }

    @classmethod
    def _build_signature(
        cls,
        jit_fn: Any,
        num_inputs: int,
        input_dtypes: list[str] | None = None,
        output_dtype: str | None = None,
    ) -> dict[str, str]:
        """Build the positional-argument signature dict for ``triton.compile``.

        Layout: ``in_ptr0, in_ptr1, …, out_ptr, n_elements`` (constexprs excluded).
        Keys are parameter *names* (strings), matching the current Triton API.
        Parameters marked ``tl.constexpr`` are excluded from the signature.

        Parameters
        ----------
        jit_fn:
            The ``@triton.jit``-decorated function.
        num_inputs:
            Number of input pointer arguments.
        input_dtypes:
            List of PyTorch dtype strings (e.g. ``["torch.float32", "torch.int32"]``).
            If ``None`` or shorter than *num_inputs*, missing entries default to
            ``"*fp32"``.
        output_dtype:
            PyTorch dtype string for the output pointer. Defaults to ``"*fp32"``.
        """
        # Collect non-constexpr parameter names from the JIT function.
        import triton.language as tl
        import inspect

        params = list(inspect.signature(jit_fn.fn).parameters.values())
        non_constexpr_names: list[str] = []
        for p in params:
            # Skip parameters annotated as tl.constexpr
            if p.annotation is tl.constexpr:
                continue
            non_constexpr_names.append(p.name)

        sig: dict[str, str] = {}
        pos = 0
        for i in range(num_inputs):
            dt_str = (
                input_dtypes[i]
                if input_dtypes is not None and i < len(input_dtypes)
                else "torch.float32"
            )
            name = non_constexpr_names[pos] if pos < len(non_constexpr_names) else f"arg{pos}"
            sig[name] = cls._TORCH_DTYPE_TO_SIG.get(dt_str, "*fp32")
            pos += 1
        # out_ptr
        out_sig = cls._TORCH_DTYPE_TO_SIG.get(output_dtype or "", "*fp32")
        name = non_constexpr_names[pos] if pos < len(non_constexpr_names) else f"arg{pos}"
        sig[name] = out_sig
        pos += 1
        # n_elements
        name = non_constexpr_names[pos] if pos < len(non_constexpr_names) else f"arg{pos}"
        sig[name] = "i32"
        return sig

    @staticmethod
    def _build_constexprs(kernel: GeneratedKernel, jit_fn: Any) -> dict[str, Any]:
        """Build the constexprs dict from metadata (e.g. ``BLOCK_SIZE``)."""
        import triton.language as tl
        import inspect

        constexprs: dict[str, Any] = {}
        params = list(inspect.signature(jit_fn.fn).parameters.values())
        for p in params:
            if p.annotation is tl.constexpr:
                # Look up the value in metadata (e.g. block_size → BLOCK_SIZE)
                key_lower = p.name.lower()
                if key_lower in kernel.metadata:
                    constexprs[p.name] = kernel.metadata[key_lower]
                elif p.name in kernel.metadata:
                    constexprs[p.name] = kernel.metadata[p.name]
        return constexprs
