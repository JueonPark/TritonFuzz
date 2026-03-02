"""KernelBuilder – constructs a Triton kernel and matching PyTorch reference.

This is the core synthesis engine.  Given a seed (via an RNG) it:

1. **Plans** the kernel configuration (number of inputs, dtypes, op count).
2. **Emits loads** for each input pointer.
3. **Emits the body** – a randomised DAG of element-wise, logic, and cast
   operations drawn from the templates in :pymod:`ops`.
4. Optionally wraps part of the body in a **for-loop** (tests LICM /
   software pipelining) or inserts **``tl.where``** branches (tests
   predicate generation).
5. **Emits the store** (epilogue).
6. **Assembles** the final source code for both Triton and PyTorch.
"""

from __future__ import annotations

import random
from typing import Optional

from tritonfuzz.generator.ops import (
    BINARY_OPS,
    COMPARISON_OPS,
    COMPARISON_THRESHOLDS,
    AtomicOpTemplate,
    OpCategory,
    OpTemplate,
    pick_atomic_op,
    pick_category,
    pick_op_from_category,
    pick_reduction_op,
)
from tritonfuzz.generator.symbol_table import SymbolTable, TensorVar
from tritonfuzz.generator.types import (
    CAST_TARGET_DTYPES,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    DType,
    pick_float_dtype,
    pick_input_dtype,
    promote,
)

# ── Tunables ─────────────────────────────────────────────────────────────

BLOCK_SIZE_CHOICES: list[int] = [64, 128, 256, 512, 1024]
N_ELEMENTS_CHOICES: list[int] = [512, 1024, 2048, 4096, 8192]

# Accumulation operators used inside generated ``for`` loops.
_LOOP_ACCUM_OPS: list[str] = ["+", "*", "-"]

# ── Dot (tl.dot) tunables ────────────────────────────────────────────────

_DOT_BLOCK_CHOICES: list[int] = [16, 32, 64]
_DOT_DIM_CHOICES: list[int] = [64, 128, 256]
_DOT_K_CHOICES: list[int] = [32, 64, 128]
_DOT_INPUT_DTYPES: list[DType] = [FLOAT16, BFLOAT16, FLOAT32]
_DOT_INPUT_WEIGHTS: list[float] = [3.0, 3.0, 1.0]


class KernelBuilder:
    """Stateful builder that constructs exactly one kernel + reference pair.

    Usage::

        builder = KernelBuilder(seed=42, rng=random.Random(42))
        triton_src, torch_src, metadata = builder.build()
    """

    def __init__(self, seed: int, rng: random.Random, *, extra_config: Optional[dict] = None) -> None:
        self.seed = seed
        self.rng = rng
        self.symtab = SymbolTable()
        self._extra_config = extra_config or {}

        # ── Configuration (decided in _plan) ─────────────────────────────
        self.num_inputs: int = 0
        self.input_dtypes: list[DType] = []
        self.use_mask: bool = True
        self.mask_use_other: bool = False
        self.block_size: int = 256
        self.n_elements: int = 1024
        self.num_body_ops: int = 4
        self.insert_loop: bool = False
        self.loop_trip_count: int = 1
        self.mix_dtypes: bool = False

        # ── Atomic store (replaces tl.store epilogue) ────────────────────
        self.use_atomic: bool = False
        self.atomic_op: Optional[AtomicOpTemplate] = None

        # ── Dot product mode (tl.dot, matmul tile) ───────────────────
        self.use_dot: bool = False
        self.dot_M: int = 0
        self.dot_N: int = 0
        self.dot_K: int = 0
        self.dot_BLOCK_M: int = 0
        self.dot_BLOCK_N: int = 0
        self.dot_BLOCK_K: int = 0
        self.dot_post_ops: int = 0
        self.dot_input_dtype: Optional[DType] = None
        self._dot_acc_name: str = ""

        # ── State (accumulated during build) ─────────────────────────────
        self._triton_body: list[str] = []
        self._torch_body: list[str] = []
        self._output_var: Optional[TensorVar] = None
        self._ops_used: list[str] = []

    # ================================================================== #
    #  Public entry point                                                  #
    # ================================================================== #

    def build(self) -> tuple[str, str, dict]:
        """Run the full synthesis pipeline.

        Returns ``(triton_source, torch_ref_source, metadata)``.
        """
        self._plan()
        self._emit_loads()
        self._emit_body()
        self._choose_output()

        # Reductions (tl.sum/max/min) compute per-block aggregates.
        # To match the PyTorch reference (which reduces globally), we
        # force single-block execution so that the block IS the full tensor.
        if any(op.startswith("reduce_") for op in self._ops_used) and not self.use_dot:
            self.n_elements = self.block_size

        triton_src = self._assemble_triton_source()
        torch_src = self._assemble_torch_ref_source()
        metadata = self._build_metadata()
        return triton_src, torch_src, metadata

    # ================================================================== #
    #  Phase 1 – Plan                                                      #
    # ================================================================== #

    def _plan(self) -> None:
        rng = self.rng

        # Number of input tensors (1‒3)
        self.num_inputs = rng.randint(1, 3)
        self.input_dtypes = [pick_input_dtype(rng) for _ in range(self.num_inputs)]

        # Masking strategy (§4.2.1)
        self.use_mask = rng.random() > 0.1        # 90 % use mask
        self.mask_use_other = rng.random() > 0.5   # 50 % add ``other=`` value

        # Block-size constexpr (§4.1.1)
        self.block_size = rng.choice(BLOCK_SIZE_CHOICES)
        self.n_elements = rng.choice(N_ELEMENTS_CHOICES)

        # Body complexity
        self.num_body_ops = rng.randint(2, 8)

        # Control-flow: for-loop (§4.3)
        self.insert_loop = rng.random() > 0.7       # 30 %
        self.loop_trip_count = rng.randint(2, 5) if self.insert_loop else 0

        # Type-stress: force diverse input dtypes (§4.2.3)
        self.mix_dtypes = rng.random() > 0.6         # 40 %
        if self.mix_dtypes and self.num_inputs >= 2:
            # Ensure at least two distinct dtypes among inputs
            attempts = 0
            while (
                len({d.triton for d in self.input_dtypes}) < 2
                and attempts < 10
            ):
                idx = rng.randint(0, self.num_inputs - 1)
                self.input_dtypes[idx] = pick_input_dtype(rng)
                attempts += 1

        # Atomic store: replace tl.store with an atomic op (15 %)
        self.use_atomic = rng.random() > 0.85
        if self.use_atomic:
            self.atomic_op = pick_atomic_op(rng)

        # Dot product kernel (10 %) — mutually exclusive with atomic
        self.use_dot = rng.random() > 0.90
        if self.use_dot:
            self.use_atomic = False
            self.atomic_op = None
            self.insert_loop = False  # K-loop is built into dot template
            self._plan_dot()

        # ── Apply extra_config overrides ─────────────────────────────────
        if "max_body_ops" in self._extra_config:
            self.num_body_ops = min(
                self.num_body_ops, int(self._extra_config["max_body_ops"]),
            )
        if "max_inputs" in self._extra_config:
            cap = int(self._extra_config["max_inputs"])
            if self.num_inputs > cap:
                self.num_inputs = cap
                self.input_dtypes = self.input_dtypes[:cap]

    def _plan_dot(self) -> None:
        """Configure matrix dimensions and block sizes for a dot kernel."""
        rng = self.rng

        # Block sizes (must be >= 16 for tensor cores)
        self.dot_BLOCK_M = rng.choice(_DOT_BLOCK_CHOICES)
        self.dot_BLOCK_N = rng.choice(_DOT_BLOCK_CHOICES)
        self.dot_BLOCK_K = rng.choice(_DOT_BLOCK_CHOICES)

        # Matrix dimensions (must be multiples of block sizes)
        m_choices = [d for d in _DOT_DIM_CHOICES if d >= self.dot_BLOCK_M]
        n_choices = [d for d in _DOT_DIM_CHOICES if d >= self.dot_BLOCK_N]
        k_choices = [d for d in _DOT_K_CHOICES if d >= self.dot_BLOCK_K]

        self.dot_M = rng.choice(m_choices) if m_choices else self.dot_BLOCK_M
        self.dot_N = rng.choice(n_choices) if n_choices else self.dot_BLOCK_N
        self.dot_K = rng.choice(k_choices) if k_choices else self.dot_BLOCK_K

        # Ensure exact divisibility
        self.dot_M = (self.dot_M // self.dot_BLOCK_M) * self.dot_BLOCK_M
        self.dot_N = (self.dot_N // self.dot_BLOCK_N) * self.dot_BLOCK_N
        self.dot_K = (self.dot_K // self.dot_BLOCK_K) * self.dot_BLOCK_K

        # Input dtype for both matrices (tl.dot requires matching types)
        self.dot_input_dtype = rng.choices(
            _DOT_INPUT_DTYPES, weights=_DOT_INPUT_WEIGHTS, k=1,
        )[0]

        # Number of element-wise post-ops on the dot result
        self.dot_post_ops = rng.randint(0, 3)

        # Override general planning for dot mode
        self.num_inputs = 2
        self.input_dtypes = [self.dot_input_dtype, self.dot_input_dtype]
        self.num_body_ops = self.dot_post_ops
        self.use_mask = False  # dimensions are aligned, no masking needed
        self.n_elements = self.dot_M * self.dot_N  # total output elements

    # ================================================================== #
    #  Phase 2 – Emit loads (§4.2.1)                                       #
    # ================================================================== #

    def _emit_loads(self) -> None:
        if self.use_dot:
            return  # Dot loads are part of the K-loop in assembly
        for i in range(self.num_inputs):
            var_name = self.symtab.fresh_name("v")  # v_0, v_1, …
            dt = self.input_dtypes[i]
            ptr = f"in_ptr{i}"

            # ── Triton load line ─────────────────────────────────────────
            mask_arg = ", mask=mask" if self.use_mask else ""
            other_arg = ""
            if self.use_mask and self.mask_use_other:
                other_val = "0.0" if dt.is_float else "0"
                other_arg = f", other={other_val}"
            triton_line = f"{var_name} = tl.load({ptr} + offsets{mask_arg}{other_arg})"

            # ── Torch reference: just alias the input tensor ─────────────
            torch_line = f"{var_name} = x{i}"

            self._triton_body.append(triton_line)
            self._torch_body.append(torch_line)
            self.symtab.add(TensorVar(var_name, dt, is_block=True))

    # ================================================================== #
    #  Phase 3 – Emit body ops (§4.2 Tensor Graph Synthesis)               #
    # ================================================================== #

    def _emit_body(self) -> None:
        if self.use_dot:
            self._emit_dot_body()
            return

        ops_emitted = 0
        loop_inserted = False

        while ops_emitted < self.num_body_ops:
            # Insert the for-loop roughly in the middle of the body.
            if (
                self.insert_loop
                and not loop_inserted
                and ops_emitted >= self.num_body_ops // 2
            ):
                self._emit_loop()
                loop_inserted = True
                ops_emitted += 1
                continue

            self._emit_one_op()
            ops_emitted += 1

    def _emit_dot_body(self) -> None:
        """Register the dot accumulator and emit optional post-ops."""
        acc_name = self.symtab.fresh_name("v")
        dot_shape = ("BLOCK_M", "BLOCK_N")
        self.symtab.add(TensorVar(acc_name, FLOAT32, is_block=True, shape=dot_shape))
        self._dot_acc_name = acc_name
        self._ops_used.append("dot")

        # Emit optional element-wise post-ops on the accumulator
        for _ in range(self.dot_post_ops):
            self._emit_one_op()

    # ── Single-op dispatcher ─────────────────────────────────────────────

    def _emit_one_op(self, indent: str = "") -> None:
        rng = self.rng
        category = pick_category(rng)

        if category == OpCategory.TYPE_CAST:
            self._emit_cast(indent)
            return

        if category == OpCategory.REDUCTION:
            self._emit_reduction(indent)
            return

        # For the LOGIC category, half the time emit ``where`` instead of
        # min/max so that we exercise predicate-based branching (§4.3).
        if category == OpCategory.LOGIC and rng.random() > 0.5:
            self._emit_where(indent)
            return

        op = pick_op_from_category(rng, category)
        if op is None:
            # Fallback to a simple binary op
            op = pick_op_from_category(rng, OpCategory.ELEMENTWISE_BINARY)
            if op is None:
                return

        if op.arity == 1:
            self._emit_unary(op, indent)
        elif op.arity == 2:
            self._emit_binary(op, indent)

    # ── Unary op emission ────────────────────────────────────────────────

    def _emit_unary(self, op: OpTemplate, indent: str = "") -> None:
        rng = self.rng

        inp = self.symtab.pick_random(rng, require_float=op.requires_float)
        if inp is None:
            # No float var available → cast first available var to float
            fallback = self.symtab.pick_random(rng)
            if fallback is None:
                return
            self._emit_cast(indent, target_dtype=FLOAT32, source_var=fallback)
            inp = self.symtab.last_var()
            if inp is None:
                return

        out_name = self.symtab.fresh_name("v")
        self._triton_body.append(f"{indent}{out_name} = {op.triton_fmt.format(inp.name)}")
        self._torch_body.append(f"{indent}{out_name} = {op.torch_fmt.format(inp.name)}")

        self.symtab.add(TensorVar(out_name, inp.dtype, is_block=True, shape=inp.shape))
        self._ops_used.append(op.name)

    # ── Binary op emission ───────────────────────────────────────────────

    def _emit_binary(self, op: OpTemplate, indent: str = "") -> None:
        rng = self.rng
        a, b = self.symtab.pick_two(rng)
        if a is None or b is None:
            return

        out_name = self.symtab.fresh_name("v")
        self._triton_body.append(
            f"{indent}{out_name} = {op.triton_fmt.format(a.name, b.name)}"
        )
        self._torch_body.append(
            f"{indent}{out_name} = {op.torch_fmt.format(a.name, b.name)}"
        )

        out_dt = promote(a.dtype, b.dtype) if op.output_dtype_rule == "promote" else a.dtype
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=a.shape))
        self._ops_used.append(op.name)

    # ── ``tl.where`` emission (§4.2.2 Logic) ────────────────────────────

    def _emit_where(self, indent: str = "") -> None:
        rng = self.rng

        # Condition variable
        cond_var = self.symtab.pick_random(rng)
        if cond_var is None:
            return

        # True / False branches – prefer same dtype for cleaner semantics
        true_var, false_var = self.symtab.pick_two(rng, prefer_same_dtype=True)
        if true_var is None or false_var is None:
            return

        cmp = rng.choice(COMPARISON_OPS)
        threshold = rng.choice(COMPARISON_THRESHOLDS)
        cond_expr = f"{cond_var.name} {cmp} {threshold}"

        out_name = self.symtab.fresh_name("v")
        self._triton_body.append(
            f"{indent}{out_name} = tl.where({cond_expr}, {true_var.name}, {false_var.name})"
        )
        self._torch_body.append(
            f"{indent}{out_name} = torch.where({cond_expr}, {true_var.name}, {false_var.name})"
        )

        out_dt = promote(true_var.dtype, false_var.dtype)
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=true_var.shape))
        self._ops_used.append("where")

    # ── Explicit type-cast emission (§4.2.3) ─────────────────────────────

    def _emit_cast(
        self,
        indent: str = "",
        target_dtype: Optional[DType] = None,
        source_var: Optional[TensorVar] = None,
    ) -> None:
        rng = self.rng
        src = source_var or self.symtab.pick_random(rng)
        if src is None:
            return

        if target_dtype is None:
            candidates = [d for d in CAST_TARGET_DTYPES if d != src.dtype]
            if not candidates:
                return
            target_dtype = rng.choice(candidates)

        out_name = self.symtab.fresh_name("v")
        self._triton_body.append(f"{indent}{out_name} = {src.name}.to({target_dtype.triton})")
        self._torch_body.append(f"{indent}{out_name} = {src.name}.to({target_dtype.torch})")

        self.symtab.add(TensorVar(out_name, target_dtype, is_block=True, shape=src.shape))
        self._ops_used.append(f"cast_to_{target_dtype.short}")

    # ── Reduction emission (tl.sum / tl.max / tl.min) ───────────────────

    def _emit_reduction(self, indent: str = "") -> None:
        """Emit a reduction followed by a broadcast-back binary op.

        Pattern::

            v_N_red = tl.sum(v_M, axis=0)      # scalar
            v_N     = v_K + v_N_red             # broadcast back to block

        This exercises the Triton compiler's reduction lowering while
        keeping the final registered variable block-shaped so that
        subsequent ops and the store epilogue work unchanged.

        Only operates on 1D block variables (``shape == ()``).  In dot
        mode all variables are 2D, so this gracefully becomes a no-op.
        """
        rng = self.rng

        # Pick a 1D block variable to reduce
        inp = self.symtab.pick_random(rng, is_block=True, require_float=True, shape=())
        if inp is None:
            inp = self.symtab.pick_random(rng, is_block=True, shape=())
        if inp is None:
            return  # no 1D blocks available (e.g. dot mode)

        red_op = pick_reduction_op(rng)

        # Emit the reduction → scalar
        scalar_name = self.symtab.fresh_name("r")
        self._triton_body.append(
            f"{indent}{scalar_name} = {red_op.triton_fmt.format(inp.name)}"
        )
        self._torch_body.append(
            f"{indent}{scalar_name} = {red_op.torch_fmt.format(inp.name)}"
        )

        # Broadcast back: combine scalar with a block variable
        block_var = self.symtab.pick_random(rng, is_block=True, shape=())
        if block_var is None:
            return

        out_name = self.symtab.fresh_name("v")
        accum_op = rng.choice(["+", "*"])
        self._triton_body.append(
            f"{indent}{out_name} = {block_var.name} {accum_op} {scalar_name}"
        )
        self._torch_body.append(
            f"{indent}{out_name} = {block_var.name} {accum_op} {scalar_name}"
        )

        out_dt = promote(block_var.dtype, inp.dtype)
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=block_var.shape))
        self._ops_used.append(red_op.name)

    # ── For-loop emission (§4.3 Control Flow) ────────────────────────────

    def _emit_loop(self) -> None:
        """Emit a ``for``-loop that accumulates into a fresh variable.

        Pattern::

            v_N = <source>
            for _loop_i in range(K):
                v_N = v_N <op> <operand>

        This exercises loop-invariant code motion (LICM) and software
        pipelining in the Triton compiler.
        """
        rng = self.rng
        src = self.symtab.pick_random(rng)
        operand = self.symtab.pick_random(rng)
        if src is None or operand is None:
            return

        acc_name = self.symtab.fresh_name("v")
        accum_op = rng.choice(_LOOP_ACCUM_OPS)
        trip = self.loop_trip_count

        # Initialise accumulator
        self._triton_body.append(f"{acc_name} = {src.name}")
        self._torch_body.append(f"{acc_name} = {src.name}")

        # Loop header
        self._triton_body.append(f"for _loop_i in range({trip}):")
        self._torch_body.append(f"for _loop_i in range({trip}):")

        # Loop body (indented)
        self._triton_body.append(
            f"    {acc_name} = {acc_name} {accum_op} {operand.name}"
        )
        self._torch_body.append(
            f"    {acc_name} = {acc_name} {accum_op} {operand.name}"
        )

        out_dt = promote(src.dtype, operand.dtype)
        self.symtab.add(TensorVar(acc_name, out_dt, is_block=True, shape=src.shape))
        self._ops_used.append(f"loop_{accum_op}")

    # ================================================================== #
    #  Phase 4 – Choose output variable                                    #
    # ================================================================== #

    def _choose_output(self) -> None:
        self._output_var = self.symtab.last_var()

    # ================================================================== #
    #  Phase 5 – Assemble final source code                                #
    # ================================================================== #

    def _assemble_triton_source(self) -> str:
        if self.use_dot:
            return self._assemble_dot_triton_source()

        seed = self.seed
        ptr_args = ", ".join(f"in_ptr{i}" for i in range(self.num_inputs))
        out_var = self._output_var.name if self._output_var else "v_0"

        lines: list[str] = [
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def triton_kernel_seed_{seed}(",
            f"    {ptr_args}, out_ptr,",
            "    n_elements,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            # ── Standard preamble (§4.1.1) ─────────────────────────────
            "    pid = tl.program_id(axis=0)",
            "    block_start = pid * BLOCK_SIZE",
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)",
        ]

        if self.use_mask:
            lines.append("    mask = offsets < n_elements")

        # ── Body (loads + computational ops) ─────────────────────────────
        for line in self._triton_body:
            lines.append(f"    {line}")

        # ── Standard epilogue (§4.1.1) ───────────────────────────────────
        if self.use_atomic and self.atomic_op is not None:
            mask_arg = ", mask=mask" if self.use_mask else ""
            lines.append(
                f"    {self.atomic_op.triton_fn}(out_ptr + offsets, {out_var}{mask_arg})"
            )
        else:
            mask_arg = ", mask=mask" if self.use_mask else ""
            lines.append(f"    tl.store(out_ptr + offsets, {out_var}{mask_arg})")

        return "\n".join(lines) + "\n"

    def _assemble_dot_triton_source(self) -> str:
        """Assemble a matmul-tile kernel using ``tl.dot``."""
        seed = self.seed
        acc = self._dot_acc_name
        out_var = self._output_var.name if self._output_var else acc

        lines: list[str] = [
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def triton_kernel_seed_{seed}(",
            "    a_ptr, b_ptr, out_ptr,",
            "    M, N, K,",
            "    stride_am, stride_ak,",
            "    stride_bk, stride_bn,",
            "    stride_cm, stride_cn,",
            "    BLOCK_M: tl.constexpr,",
            "    BLOCK_N: tl.constexpr,",
            "    BLOCK_K: tl.constexpr,",
            "):",
            "    pid_m = tl.program_id(0)",
            "    pid_n = tl.program_id(1)",
            "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"    {acc} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
            "    for k_start in range(0, K, BLOCK_K):",
            "        offs_k = k_start + tl.arange(0, BLOCK_K)",
            "        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)",
            "        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)",
            f"        {acc} += tl.dot(a, b)",
        ]

        # Post-ops from body
        for line in self._triton_body:
            lines.append(f"    {line}")

        # Store
        lines.extend([
            "    c_ptrs = out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn",
            f"    tl.store(c_ptrs, {out_var})",
        ])

        return "\n".join(lines) + "\n"

    def _assemble_torch_ref_source(self) -> str:
        if self.use_dot:
            return self._assemble_dot_torch_ref_source()

        seed = self.seed
        input_args = ", ".join(f"x{i}" for i in range(self.num_inputs))
        out_var = self._output_var.name if self._output_var else "v_0"

        lines: list[str] = [
            "import torch",
            "",
            "",
            f"def torch_ref_seed_{seed}({input_args}):",
        ]

        for line in self._torch_body:
            lines.append(f"    {line}")

        lines.append(f"    return {out_var}")

        return "\n".join(lines) + "\n"

    def _assemble_dot_torch_ref_source(self) -> str:
        """Assemble the PyTorch reference for a dot kernel."""
        seed = self.seed
        acc = self._dot_acc_name
        out_var = self._output_var.name if self._output_var else acc

        lines: list[str] = [
            "import torch",
            "",
            "",
            f"def torch_ref_seed_{seed}(a, b):",
            f"    {acc} = torch.matmul(a.float(), b.float())",
        ]

        for line in self._torch_body:
            lines.append(f"    {line}")

        lines.append(f"    return {out_var}")

        return "\n".join(lines) + "\n"

    # ================================================================== #
    #  Metadata                                                            #
    # ================================================================== #

    def _build_metadata(self) -> dict:
        meta = {
            "generator_version": "0.4.0",
            "seed": self.seed,
            "num_inputs": self.num_inputs,
            "input_dtypes": [d.torch for d in self.input_dtypes],
            "output_dtype": (
                self._output_var.dtype.torch if self._output_var else "torch.float32"
            ),
            "block_size": self.block_size,
            "n_elements": self.n_elements,
            "kernel_fn_name": f"triton_kernel_seed_{self.seed}",
            "ref_fn_name": f"torch_ref_seed_{self.seed}",
            "num_body_ops": self.num_body_ops,
            "ops_used": list(self._ops_used),
            "use_mask": self.use_mask,
            "use_other": self.mask_use_other,
            "has_loop": self.insert_loop,
            "loop_trip_count": self.loop_trip_count,
            "mix_dtypes": self.mix_dtypes,
            "has_atomics": self.use_atomic,
            "use_dot": self.use_dot,
        }
        if self.use_atomic and self.atomic_op is not None:
            meta["atomic_op"] = self.atomic_op.name
            meta["output_init"] = self.atomic_op.output_init
            self._ops_used.append(self.atomic_op.name)
            meta["ops_used"] = list(self._ops_used)
        if self.use_dot:
            meta["dot_M"] = self.dot_M
            meta["dot_N"] = self.dot_N
            meta["dot_K"] = self.dot_K
            meta["dot_BLOCK_M"] = self.dot_BLOCK_M
            meta["dot_BLOCK_N"] = self.dot_BLOCK_N
            meta["dot_BLOCK_K"] = self.dot_BLOCK_K
            meta["input_shapes"] = [
                [self.dot_M, self.dot_K],
                [self.dot_K, self.dot_N],
            ]
            meta["output_shape"] = [self.dot_M, self.dot_N]
            meta["output_dtype"] = "torch.float32"  # dot accumulator is always fp32
        return meta
