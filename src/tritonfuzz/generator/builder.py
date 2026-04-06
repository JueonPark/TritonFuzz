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
from tritonfuzz.generator.symbol_table import (
    SymbolTable,
    TensorVar,
    flatten_shape_expr,
    shapes_same_numel,
    transpose_shape,
)
from tritonfuzz.generator.types import (
    CAST_TARGET_DTYPES,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    INT32,
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

# ── Scalar-condition tunables (for if/else and while) ────────────────────

# Reduction functions used to derive scalar conditions from block tensors.
_COND_REDUCTION_TRITON: list[str] = [
    "tl.sum({0}, axis=0)",
    "tl.max({0}, axis=0)",
    "tl.min({0}, axis=0)",
]
_COND_REDUCTION_TORCH: list[str] = [
    "torch.sum({0})",
    "torch.max({0})",
    "torch.min({0})",
]
_COND_THRESHOLDS: list[str] = ["0.0", "0.5", "1.0", "-0.5", "-1.0"]
_COND_COMPARISONS: list[str] = [">", "<", ">=", "<="]

# Maximum iteration counts for bounded while-loops.
_WHILE_MAX_ITER_CHOICES: list[int] = [4, 8, 16]

# ── Nested control-flow patterns ─────────────────────────────────────────
_NESTED_CF_PATTERNS: list[str] = ["if_in_if", "if_in_while", "while_in_if"]

# ── Dot (tl.dot) tunables ────────────────────────────────────────────────

_DOT_BLOCK_CHOICES: list[int] = [16, 32, 64]
_DOT_DIM_CHOICES: list[int] = [64, 128, 256]
_DOT_K_CHOICES: list[int] = [32, 64, 128]
_DOT_INPUT_DTYPES: list[DType] = [FLOAT16, BFLOAT16, FLOAT32]
_DOT_INPUT_WEIGHTS: list[float] = [3.0, 3.0, 1.0]

# ── Layout-conversion op weights (for _emit_layout_op dispatcher) ────

_LAYOUT_OP_CHOICES: list[str] = [
    "trans",
    "reshape_flatten",
    "reshape_unflatten",
    "expand_broadcast",
]
_LAYOUT_OP_WEIGHTS: list[float] = [3.0, 2.5, 2.5, 2.0]

# ── Block-pointer tunables ───────────────────────────────────────────

# Block-pointer mode requires n_elements to be an exact multiple of
# BLOCK_SIZE so that ``tl.advance`` steps are uniform.  We restrict
# the choices accordingly.
_BLOCK_PTR_BLOCK_SIZES: list[int] = [64, 128, 256]

# ── Gather/scatter tunables ──────────────────────────────────────────

# Maximum value for randomly generated indices (as a fraction of
# n_elements) to keep accesses in-bounds.
_GATHER_N_ELEMENTS_CHOICES: list[int] = [512, 1024, 2048, 4096]

# ── Register-pressure stress tunables ────────────────────────────────

# Block sizes well past 1024 inflate per-thread register consumption
# and force the MLIR pipeline into aggressive spilling / shared-memory
# allocation paths.
_REG_PRESSURE_BLOCK_SIZES: list[int] = [1024, 2048, 4096]
_REG_PRESSURE_BODY_OPS_RANGE: tuple[int, int] = (10, 20)
_REG_PRESSURE_LOOP_TRIP_RANGE: tuple[int, int] = (4, 16)
_REG_PRESSURE_NUM_WARPS: list[int] = [1, 2, 4, 8, 16]
_REG_PRESSURE_NUM_STAGES: list[int] = [1, 2, 3, 4, 5]
# Parallel accumulation chain counts (all live simultaneously)
_REG_PRESSURE_CHAIN_COUNTS: list[int] = [2, 3, 4]


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

        # ── Divergent control flow (if/else, while, nested) ────────
        self.insert_if_else: bool = False
        self.if_else_branch_ops: int = 1
        self.insert_while_loop: bool = False
        self.while_max_iter: int = 4
        self.insert_nested_cf: bool = False
        self.nested_cf_pattern: str = ""

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

        # ── Layout-conversion stress (dot mode only) ─────────────────
        self.dot_layout_ops: int = 0
        self.dot_use_layout_stress: bool = False
        self.dot_square_blocks: bool = False

        # ── Multi-dot mode (chained tl.dot with layout conversion) ───
        self.use_multi_dot: bool = False
        self.multi_dot_P: int = 0
        self.multi_dot_BLOCK_P: int = 0
        self._multi_dot_acc2_name: str = ""

        # ── Block-pointer mode (tl.make_block_ptr + tl.advance) ──────
        self.use_block_ptr: bool = False

        # ── Gather/scatter mode (indirect memory access) ─────────────
        self.use_gather: bool = False
        self.gather_data_inputs: int = 0  # how many data inputs use gather

        # ── Register-pressure stress mode ────────────────────────────────
        self.use_reg_pressure: bool = False
        self.reg_pressure_num_warps: int = 4
        self.reg_pressure_num_stages: int = 2
        self.reg_pressure_parallel_chains: int = 0

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
        # This also applies to control-flow constructs whose conditions
        # use reductions (if/else, while, nested CF).
        has_reduction_cf = any(
            op in self._ops_used
            for op in ("if_else", "while_loop", "nested_if_in_if",
                        "loop_with_branch", "while_in_if")
        )
        if (any(op.startswith("reduce_") for op in self._ops_used) or has_reduction_cf) and not self.use_dot:
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

        # Block-pointer mode (15 % of NON-dot, NON-atomic kernels)
        # Uses tl.make_block_ptr + tl.load(block_ptr) + tl.advance
        # instead of explicit offset computation.  Mutually exclusive
        # with dot mode and atomics (different store pattern).
        if not self.use_dot and not self.use_atomic:
            self.use_block_ptr = rng.random() > 0.85
            if self.use_block_ptr:
                # Block-pointer needs exact divisibility
                self.block_size = rng.choice(_BLOCK_PTR_BLOCK_SIZES)
                # Force n_elements to be an exact multiple of block_size
                multiplier = rng.choice([4, 8, 16, 32])
                self.n_elements = self.block_size * multiplier
                self.use_mask = False  # no masking needed with exact divisibility

        # Gather/scatter mode (12 % of NON-dot kernels)
        # Adds an index tensor as the first input; subsequent data loads
        # use tl.load(data_ptr + idx) where idx is loaded from the index
        # tensor, exercising unstructured gather instruction generation.
        # Compatible with block_ptr mode and atomics.
        if not self.use_dot:
            self.use_gather = rng.random() > 0.88
            if self.use_gather:
                # Require at least 1 data input to gather through
                self.num_inputs = max(self.num_inputs, 2)
                # How many of the data inputs use gathered indexing (1 or all)
                self.gather_data_inputs = rng.randint(1, self.num_inputs - 1)
                # Keep n_elements reasonable for gather
                self.n_elements = rng.choice(_GATHER_N_ELEMENTS_CHOICES)
                # First input is the index tensor (int32)
                self.input_dtypes[0] = INT32
                # Ensure we have enough dtype entries
                while len(self.input_dtypes) < self.num_inputs:
                    self.input_dtypes.append(pick_input_dtype(rng))
                # Disable block_ptr if gather is on — they are different
                # pointer strategies on the load side
                self.use_block_ptr = False

        # ── Register-pressure stress (15 % of NON-dot kernels) ──────
        # Bumps BLOCK_SIZE well past 1024, generates many temporary
        # variables, and forces high loop-unrolling to inflate register
        # pressure and trigger spilling / shared-memory allocation paths.
        if not self.use_dot:
            self.use_reg_pressure = rng.random() > 0.85
            if self.use_reg_pressure:
                self.block_size = rng.choice(_REG_PRESSURE_BLOCK_SIZES)
                # n_elements must be a multiple of block_size
                multiplier = rng.choice([2, 4, 8])
                self.n_elements = self.block_size * multiplier
                # Many body ops ⇒ lots of live temporaries
                self.num_body_ops = rng.randint(*_REG_PRESSURE_BODY_OPS_RANGE)
                # Always insert a loop with high trip count
                self.insert_loop = True
                self.loop_trip_count = rng.randint(*_REG_PRESSURE_LOOP_TRIP_RANGE)
                # Force 3 inputs for maximum data-flow diversity
                self.num_inputs = 3
                while len(self.input_dtypes) < self.num_inputs:
                    self.input_dtypes.append(pick_input_dtype(rng))
                # Explicit num_warps / num_stages for compiler stress
                self.reg_pressure_num_warps = rng.choice(_REG_PRESSURE_NUM_WARPS)
                self.reg_pressure_num_stages = rng.choice(_REG_PRESSURE_NUM_STAGES)
                # Parallel accumulation chains (keep many regs live)
                self.reg_pressure_parallel_chains = rng.choice(
                    _REG_PRESSURE_CHAIN_COUNTS
                )
                # Disable block_ptr — its structured loads interact
                # awkwardly with extreme block sizes
                self.use_block_ptr = False

        # ── Divergent control flow (not in dot mode) ─────────────────
        if not self.use_dot:
            # Nested control flow (10 %) — subsumes if/else and while
            self.insert_nested_cf = rng.random() > 0.90
            if self.insert_nested_cf:
                self.nested_cf_pattern = rng.choice(_NESTED_CF_PATTERNS)
                self.if_else_branch_ops = rng.randint(1, 3)
                self.while_max_iter = rng.choice(_WHILE_MAX_ITER_CHOICES)
            else:
                # If/else branches (25 %)
                self.insert_if_else = rng.random() > 0.75
                if self.insert_if_else:
                    self.if_else_branch_ops = rng.randint(1, 3)

                # Data-dependent while loop (20 %)
                self.insert_while_loop = rng.random() > 0.80
                if self.insert_while_loop:
                    self.while_max_iter = rng.choice(_WHILE_MAX_ITER_CHOICES)

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

        # Layout-conversion stress (30 % of dot kernels)
        self.dot_use_layout_stress = rng.random() > 0.70
        if self.dot_use_layout_stress:
            self.dot_layout_ops = rng.randint(1, 3)
            # Ensure enough post-ops so layout ops sit between compute ops
            self.dot_post_ops = max(self.dot_post_ops, 2)
            # Square blocks needed for reshape flatten/unflatten round-trips
            self.dot_square_blocks = rng.random() > 0.5
            if self.dot_square_blocks:
                common = rng.choice(_DOT_BLOCK_CHOICES)
                self.dot_BLOCK_M = common
                self.dot_BLOCK_N = common
                # Re-align M, N to the new common block size
                m_choices = [d for d in _DOT_DIM_CHOICES if d >= common]
                n_choices = [d for d in _DOT_DIM_CHOICES if d >= common]
                self.dot_M = rng.choice(m_choices) if m_choices else common
                self.dot_N = rng.choice(n_choices) if n_choices else common
                self.dot_M = (self.dot_M // common) * common
                self.dot_N = (self.dot_N // common) * common
        else:
            self.dot_layout_ops = 0

        # Override general planning for dot mode
        self.num_inputs = 2
        self.input_dtypes = [self.dot_input_dtype, self.dot_input_dtype]
        self.num_body_ops = self.dot_post_ops
        self.use_mask = False  # dimensions are aligned, no masking needed
        self.n_elements = self.dot_M * self.dot_N  # total output elements

        # Multi-dot: chained tl.dot with transpose between them (5 %)
        # Requires square blocks: BLOCK_M == BLOCK_N so that the
        # transposed accumulator can be used as input to a second tl.dot.
        # Pattern: acc1 = dot(A, B) → trans(acc1) → acc2 = dot(trans, C)
        if rng.random() > 0.95 and self.dot_BLOCK_M == self.dot_BLOCK_N:
            self.use_multi_dot = True
            # Third matrix C has shape (N, P) → dot(acc1.T, C) gives (M, P)
            # For simplicity, use P == N and BLOCK_P == BLOCK_N
            self.multi_dot_P = self.dot_N
            self.multi_dot_BLOCK_P = self.dot_BLOCK_N
            self.num_inputs = 3
            self.input_dtypes = [self.dot_input_dtype] * 3
            self.n_elements = self.dot_M * self.multi_dot_P
            # Force layout stress on for multi-dot
            self.dot_use_layout_stress = True
            self.dot_layout_ops = max(self.dot_layout_ops, 1)

    # ================================================================== #
    #  Phase 2 – Emit loads (§4.2.1)                                       #
    # ================================================================== #

    def _emit_loads(self) -> None:
        if self.use_dot:
            return  # Dot loads are part of the K-loop in assembly
        if self.use_gather:
            self._emit_gather_loads()
            return
        if self.use_block_ptr:
            self._emit_block_ptr_loads()
            return
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

    def _emit_block_ptr_loads(self) -> None:
        """Emit loads using ``tl.make_block_ptr`` / ``tl.load(block_ptr)``.

        For each input, creates a block pointer in the assembly preamble
        and loads via the structured-memory path.  The body lines emitted
        here are just the ``tl.load`` calls; the ``make_block_ptr`` setup
        lives in the assembly template.
        """
        for i in range(self.num_inputs):
            var_name = self.symtab.fresh_name("v")
            dt = self.input_dtypes[i]
            # Block-pointer load: no explicit offsets, no mask
            triton_line = f"{var_name} = tl.load(in_blk_ptr{i})"
            torch_line = f"{var_name} = x{i}"

            self._triton_body.append(triton_line)
            self._torch_body.append(torch_line)
            self.symtab.add(TensorVar(var_name, dt, is_block=True))
        self._ops_used.append("block_ptr_load")

    def _emit_gather_loads(self) -> None:
        """Emit loads with indirect (gather) indexing.

        Pattern — input 0 is an index tensor, inputs 1..N are data::

            idx   = tl.load(idx_ptr + offsets, mask=mask)
            v_0   = x0                      # torch ref
            v_1   = tl.load(in_ptr1 + idx)  # gathered data

        The index tensor contains pre-clamped int32 offsets so the
        gather is always in-bounds.  Data inputs beyond
        ``gather_data_inputs`` fall back to standard linear loads.
        """
        rng = self.rng
        mask_arg = ", mask=mask" if self.use_mask else ""
        other_arg_int = ", other=0" if self.use_mask else ""

        # Load the index tensor (input 0 is always the index)
        idx_name = self.symtab.fresh_name("idx")
        self._triton_body.append(
            f"{idx_name} = tl.load(idx_ptr + offsets{mask_arg}{other_arg_int})"
        )
        self._torch_body.append(
            f"{idx_name} = x_idx"
        )
        # idx is int32, not a compute variable — don't add to symtab
        # (it's only used for addressing)

        # Load data inputs via gather
        for i in range(self.num_inputs - 1):
            data_idx = i + 1  # skip index input (input 0)
            var_name = self.symtab.fresh_name("v")
            dt = self.input_dtypes[data_idx]

            if i < self.gather_data_inputs:
                # Gathered load — use idx as the offset
                other_arg = ""
                if self.use_mask:
                    other_val = "0.0" if dt.is_float else "0"
                    other_arg = f", other={other_val}"
                triton_line = f"{var_name} = tl.load(in_ptr{data_idx} + {idx_name}{mask_arg}{other_arg})"
                torch_line = f"{var_name} = x{data_idx}[{idx_name}]"
            else:
                # Standard linear load for remaining inputs
                other_arg = ""
                if self.use_mask and self.mask_use_other:
                    other_val = "0.0" if dt.is_float else "0"
                    other_arg = f", other={other_val}"
                triton_line = f"{var_name} = tl.load(in_ptr{data_idx} + offsets{mask_arg}{other_arg})"
                torch_line = f"{var_name} = x{data_idx}"

            self._triton_body.append(triton_line)
            self._torch_body.append(torch_line)
            self.symtab.add(TensorVar(var_name, dt, is_block=True))

        self._ops_used.append("gather_load")

    # ================================================================== #
    #  Phase 3 – Emit body ops (§4.2 Tensor Graph Synthesis)               #
    # ================================================================== #

    def _emit_body(self) -> None:
        if self.use_dot:
            self._emit_dot_body()
            return

        ops_emitted = 0
        loop_inserted = False
        if_else_inserted = False
        while_inserted = False
        nested_cf_inserted = False
        parallel_chains_inserted = False

        while ops_emitted < self.num_body_ops:
            # Insert nested control flow roughly at 1/3 of body.
            if (
                self.insert_nested_cf
                and not nested_cf_inserted
                and ops_emitted >= self.num_body_ops // 3
            ):
                self._emit_nested_cf()
                nested_cf_inserted = True
                ops_emitted += 1
                continue

            # Insert if/else roughly at 1/3 of body.
            if (
                self.insert_if_else
                and not if_else_inserted
                and not self.insert_nested_cf
                and ops_emitted >= self.num_body_ops // 3
            ):
                self._emit_if_else()
                if_else_inserted = True
                ops_emitted += 1
                continue

            # Insert while-loop roughly at 2/3 of body.
            if (
                self.insert_while_loop
                and not while_inserted
                and not self.insert_nested_cf
                and ops_emitted >= (self.num_body_ops * 2) // 3
            ):
                self._emit_while_loop()
                while_inserted = True
                ops_emitted += 1
                continue

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

            # Parallel accumulation chains for register pressure stress.
            # Placed at ~2/3 of body so many earlier variables are already
            # live; the chains keep additional registers alive across a
            # long code span.
            if (
                self.use_reg_pressure
                and self.reg_pressure_parallel_chains > 0
                and not parallel_chains_inserted
                and ops_emitted >= (self.num_body_ops * 2) // 3
            ):
                self._emit_parallel_chains()
                parallel_chains_inserted = True
                ops_emitted += 1
                continue

            self._emit_one_op()
            ops_emitted += 1

    def _emit_dot_body(self) -> None:
        """Register the dot accumulator and emit optional post-ops.

        When layout-stress mode is active, layout-conversion operations
        (transpose, reshape flatten/unflatten, expand+broadcast) are
        interspersed among the element-wise post-ops to force the Triton
        compiler to emit intermediate layout conversions between
        Blocked / Shared / DotOperand / Slice encodings.
        """
        if self.use_multi_dot:
            self._emit_multi_dot_body()
            return

        acc_name = self.symtab.fresh_name("v")
        dot_shape = ("BLOCK_M", "BLOCK_N")
        self.symtab.add(TensorVar(acc_name, FLOAT32, is_block=True, shape=dot_shape))
        self._dot_acc_name = acc_name
        self._ops_used.append("dot")

        if not self.dot_use_layout_stress:
            # Simple path: just element-wise post-ops
            for _ in range(self.dot_post_ops):
                self._emit_one_op()
            return

        # Layout-stress path: interleave layout ops among post-ops.
        # Pattern: 1-2 post-ops, layout-op, 1-2 post-ops, layout-op, …
        layout_ops_remaining = self.dot_layout_ops
        post_ops_remaining = self.dot_post_ops
        rng = self.rng

        while post_ops_remaining > 0 or layout_ops_remaining > 0:
            # Emit 1-2 element-wise post-ops
            chunk = min(rng.randint(1, 2), post_ops_remaining)
            for _ in range(chunk):
                self._emit_one_op()
                post_ops_remaining -= 1

            # Emit a layout op (if any remaining)
            if layout_ops_remaining > 0:
                self._emit_layout_op()
                layout_ops_remaining -= 1

        # Ensure the final variable has the correct shape for the store
        self._ensure_output_shape(dot_shape)

    def _emit_multi_dot_body(self) -> None:
        """Emit a chained dot pattern: dot → trans → dot.

        Pattern::

            acc1   = tl.dot(A_tile, B_tile)    # (BLOCK_M, BLOCK_N)
            acc1_t = tl.trans(acc1)             # (BLOCK_N, BLOCK_M)
            acc2   = tl.dot(acc1_t, C_tile)    # (BLOCK_N, BLOCK_P)

        Since BLOCK_M == BLOCK_N (square), the transposed acc1 has shape
        (BLOCK_M, BLOCK_M) and can be used as a valid dot-LHS.  The
        output is (BLOCK_M, BLOCK_P) where BLOCK_P == BLOCK_N.

        This forces the compiler to handle DotOperand → Shared → Blocked
        layout conversions across the transpose boundary.
        """
        dot_shape = ("BLOCK_M", "BLOCK_N")

        # First dot accumulator (handled by assembly template)
        acc1_name = self.symtab.fresh_name("v")
        self.symtab.add(TensorVar(acc1_name, FLOAT32, is_block=True, shape=dot_shape))
        self._dot_acc_name = acc1_name
        self._ops_used.append("dot")

        # Transpose: (BLOCK_M, BLOCK_N) → (BLOCK_N, BLOCK_M)
        trans_name = self.symtab.fresh_name("v")
        self._triton_body.append(f"{trans_name} = tl.trans({acc1_name})")
        self._torch_body.append(f"{trans_name} = {acc1_name}.T")
        trans_shape = transpose_shape(dot_shape)
        self.symtab.add(TensorVar(trans_name, FLOAT32, is_block=True, shape=trans_shape))
        self._ops_used.append("trans")

        # Second dot: emit as body code (not in assembly K-loop)
        # acc2 = tl.dot(acc1_t, C_tile) where C_tile loads are inlined
        acc2_name = self.symtab.fresh_name("v")
        # This second dot is represented as element-wise ops in the body;
        # the actual second tl.dot call is assembled in the template.
        self.symtab.add(TensorVar(acc2_name, FLOAT32, is_block=True, shape=dot_shape))
        self._multi_dot_acc2_name = acc2_name
        self._ops_used.append("dot")

        # Optional post-ops on the final accumulator
        layout_ops_remaining = self.dot_layout_ops
        post_ops_remaining = self.dot_post_ops
        rng = self.rng

        while post_ops_remaining > 0 or layout_ops_remaining > 0:
            chunk = min(rng.randint(1, 2), post_ops_remaining)
            for _ in range(chunk):
                self._emit_one_op()
                post_ops_remaining -= 1
            if layout_ops_remaining > 0:
                self._emit_layout_op()
                layout_ops_remaining -= 1

        self._ensure_output_shape(dot_shape)

    # ── Single-op dispatcher ─────────────────────────────────────────────

    def _emit_one_op(self, indent: str = "") -> None:
        rng = self.rng
        category = pick_category(rng)

        if category == OpCategory.LAYOUT_CONVERT:
            # Layout ops only useful in dot mode (2D tensors).
            if self.use_dot:
                self._emit_layout_op(indent)
            # In 1D mode, fall-through to a binary op instead.
            else:
                category = OpCategory.ELEMENTWISE_BINARY

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

        # In dot mode with layout ops, tensors may have different shapes
        # (e.g. 2-D vs 1-D after flatten).  Ensure both operands share
        # the same shape to avoid Triton shape-mismatch errors.
        if a.shape != b.shape:
            b = self.symtab.pick_random(rng, shape=a.shape)
            if b is None:
                b = a  # self-op as last resort

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

        # Ensure all three operands share the same shape
        if cond_var.shape != true_var.shape:
            true_var = self.symtab.pick_random(rng, shape=cond_var.shape)
            if true_var is None:
                return
        if false_var.shape != true_var.shape:
            false_var = self.symtab.pick_random(rng, shape=true_var.shape)
            if false_var is None:
                false_var = true_var

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

    # ── Layout-conversion emission (dot mode only) ───────────────────────

    def _emit_layout_op(self, indent: str = "") -> None:
        """Dispatch to a random layout-conversion operation.

        Layout ops only make sense when 2-D block tensors are present
        (dot mode).  Each sub-emitter is responsible for gracefully
        returning if no suitable operand exists.
        """
        rng = self.rng
        choice = rng.choices(_LAYOUT_OP_CHOICES, weights=_LAYOUT_OP_WEIGHTS, k=1)[0]

        if choice == "trans":
            self._emit_trans(indent)
        elif choice == "reshape_flatten":
            self._emit_reshape_flatten(indent)
        elif choice == "reshape_unflatten":
            self._emit_reshape_unflatten(indent)
        elif choice == "expand_broadcast":
            self._emit_expand_broadcast(indent)

    def _emit_trans(self, indent: str = "") -> None:
        """Emit ``tl.trans(x)`` on a 2-D block tensor.

        Triton: ``v_N = tl.trans(v_M)``
        Torch:  ``v_N = v_M.T``

        Forces a layout conversion between Blocked and Shared encodings
        in the Triton MLIR pipeline.
        """
        rng = self.rng
        inp = self.symtab.pick_random_by_ndim(rng, 2, require_float=True)
        if inp is None:
            inp = self.symtab.pick_random_by_ndim(rng, 2)
        if inp is None:
            return

        out_name = self.symtab.fresh_name("v")

        # Randomly choose between tl.trans() and tl.permute() for more
        # MLIR path coverage.
        if rng.random() > 0.5:
            self._triton_body.append(f"{indent}{out_name} = tl.trans({inp.name})")
        else:
            self._triton_body.append(f"{indent}{out_name} = tl.permute({inp.name}, (1, 0))")
        self._torch_body.append(f"{indent}{out_name} = {inp.name}.T")

        out_shape = transpose_shape(inp.shape)
        self.symtab.add(TensorVar(out_name, inp.dtype, is_block=True, shape=out_shape))
        self._ops_used.append("trans")

    def _emit_reshape_flatten(self, indent: str = "") -> None:
        """Flatten a 2-D block tensor to 1-D via ``tl.reshape``.

        Triton: ``v_N = tl.reshape(v_M, (BLOCK_M * BLOCK_N,))``
        Torch:  ``v_N = v_M.reshape(M * N)``

        Randomly toggles ``can_reorder=True`` to stress both MLIR
        lowering paths.
        """
        rng = self.rng
        inp = self.symtab.pick_random_by_ndim(rng, 2)
        if inp is None:
            return

        numel_expr = flatten_shape_expr(inp.shape)  # e.g. "BLOCK_M * BLOCK_N"
        flat_shape = (numel_expr,)                    # 1-D symbolic shape

        out_name = self.symtab.fresh_name("v")
        can_reorder = rng.choice(["True", "False"])
        self._triton_body.append(
            f"{indent}{out_name} = tl.reshape({inp.name}, ({numel_expr},), can_reorder={can_reorder})"
        )
        self._torch_body.append(
            f"{indent}{out_name} = {inp.name}.reshape(-1)"
        )

        self.symtab.add(TensorVar(out_name, inp.dtype, is_block=True, shape=flat_shape))
        self._ops_used.append("reshape_flatten")

    def _emit_reshape_unflatten(self, indent: str = "") -> None:
        """Unflatten a 1-D block tensor back to 2-D via ``tl.reshape``.

        Only operates on 1-D tensors whose symbolic element count matches
        ``BLOCK_M * BLOCK_N`` (i.e. produced by a prior flatten).

        Triton: ``v_N = tl.reshape(v_M, (BLOCK_M, BLOCK_N))``
        Torch:  ``v_N = v_M.reshape(M, N)``
        """
        rng = self.rng
        dot_shape = ("BLOCK_M", "BLOCK_N")
        flat_numel = flatten_shape_expr(dot_shape)  # "BLOCK_M * BLOCK_N"

        # Find a 1-D variable whose shape is the flattened form
        candidates = [
            v for v in self.symtab.all_vars()
            if v.is_block and len(v.shape) == 1 and shapes_same_numel(v.shape, dot_shape)
        ]
        if not candidates:
            return

        inp = rng.choice(candidates)
        out_name = self.symtab.fresh_name("v")
        can_reorder = rng.choice(["True", "False"])
        self._triton_body.append(
            f"{indent}{out_name} = tl.reshape({inp.name}, (BLOCK_M, BLOCK_N), can_reorder={can_reorder})"
        )
        self._torch_body.append(
            f"{indent}{out_name} = {inp.name}.reshape(M, N)"
        )

        self.symtab.add(TensorVar(out_name, inp.dtype, is_block=True, shape=dot_shape))
        self._ops_used.append("reshape_unflatten")

    def _emit_expand_broadcast(self, indent: str = "") -> None:
        """Reduce → expand_dims → broadcast_to on a 2-D block tensor.

        Three-step pattern that forces multiple layout conversions::

            r   = tl.sum(x, axis=1)             # (BLOCK_M, BLOCK_N) → (BLOCK_M,)
            e   = tl.expand_dims(r, 1)          # (BLOCK_M,) → (BLOCK_M, 1)
            out = tl.broadcast_to(e, (BLOCK_M, BLOCK_N))  # → (BLOCK_M, BLOCK_N)

        PyTorch equivalent::

            r   = x.sum(dim=1)
            out = r.unsqueeze(1).expand(M, N)
        """
        rng = self.rng
        inp = self.symtab.pick_random_by_ndim(rng, 2, require_float=True)
        if inp is None:
            inp = self.symtab.pick_random_by_ndim(rng, 2)
        if inp is None:
            return

        # Step 1: reduce along axis 1 → 1-D
        red_name = self.symtab.fresh_name("r")
        red_fn_triton = rng.choice(["tl.sum", "tl.max", "tl.min"])
        red_fn_torch_map = {"tl.sum": "torch.sum", "tl.max": "torch.amax", "tl.min": "torch.amin"}
        red_fn_torch = red_fn_torch_map[red_fn_triton]

        self._triton_body.append(
            f"{indent}{red_name} = {red_fn_triton}({inp.name}, axis=1)"
        )
        self._torch_body.append(
            f"{indent}{red_name} = {red_fn_torch}({inp.name}, dim=1)"
        )

        # Step 2: expand_dims on the reduced result → 2-D with dim-1 column
        exp_name = self.symtab.fresh_name("v")
        self._triton_body.append(
            f"{indent}{exp_name} = tl.expand_dims({red_name}, 1)"
        )
        self._torch_body.append(
            f"{indent}{exp_name} = {red_name}.unsqueeze(1)"
        )

        # Step 3: broadcast back to original 2-D shape
        out_name = self.symtab.fresh_name("v")
        self._triton_body.append(
            f"{indent}{out_name} = tl.broadcast_to({exp_name}, ({inp.shape[0]}, {inp.shape[1]}))"
        )
        self._torch_body.append(
            f"{indent}{out_name} = {exp_name}.expand({inp.shape[0]}, {inp.shape[1]})"
        )

        self.symtab.add(TensorVar(out_name, inp.dtype, is_block=True, shape=inp.shape))
        self._ops_used.append("expand_broadcast")

    def _ensure_output_shape(self, target_shape: tuple[str, ...]) -> None:
        """If the last variable's shape differs from *target_shape*,
        emit a corrective ``tl.reshape`` or ``tl.trans`` so that the
        store epilogue can use the expected indexing pattern.
        """
        last = self.symtab.last_var()
        if last is None or last.shape == target_shape:
            return

        out_name = self.symtab.fresh_name("v")

        # Case 1: transposed 2-D → just transpose back
        if transpose_shape(last.shape) == target_shape and len(last.shape) == 2:
            self._triton_body.append(f"{out_name} = tl.trans({last.name})")
            self._torch_body.append(f"{out_name} = {last.name}.T")
            self.symtab.add(TensorVar(out_name, last.dtype, is_block=True, shape=target_shape))
            self._ops_used.append("trans_fixup")
            return

        # Case 2: flattened 1-D → reshape back to 2-D
        if shapes_same_numel(last.shape, target_shape):
            target_dims = ", ".join(target_shape)
            self._triton_body.append(
                f"{out_name} = tl.reshape({last.name}, ({target_dims},))"
            )
            if len(target_shape) == 2:
                self._torch_body.append(
                    f"{out_name} = {last.name}.reshape({target_shape[0]}, {target_shape[1]})"
                )
            else:
                self._torch_body.append(
                    f"{out_name} = {last.name}.reshape(-1)"
                )
            self.symtab.add(TensorVar(out_name, last.dtype, is_block=True, shape=target_shape))
            self._ops_used.append("reshape_fixup")
            return

        # Case 3: incompatible — shouldn't happen with well-planned ops.
        # As a safety net, reuse the last variable without shape change.

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

    # ── Parallel accumulation chains (register-pressure stress) ──────────

    def _emit_parallel_chains(self) -> None:
        """Emit *N* independent computation chains whose results are
        combined at the end.

        Each chain picks a different starting variable and performs 2-4
        element-wise ops, keeping its own set of temporaries live.
        Because all chain outputs are combined in a final reduction,
        the compiler cannot release any chain's registers until after
        the merge — artificially maximising register pressure.

        Pattern::

            chain_0 = f(v_A)      # 2-4 ops
            chain_1 = g(v_B)      # 2-4 ops, independent of chain_0
            …
            v_out = chain_0 + chain_1 + …
        """
        rng = self.rng
        n_chains = self.reg_pressure_parallel_chains

        chain_vars: list[TensorVar] = []
        for chain_idx in range(n_chains):
            # Pick a starting variable for this chain
            start = self.symtab.pick_random(rng, is_block=True, shape=())
            if start is None:
                continue

            chain_name = self.symtab.fresh_name("chain")
            self._triton_body.append(f"{chain_name} = {start.name}")
            self._torch_body.append(f"{chain_name} = {start.name}")
            self.symtab.add(TensorVar(chain_name, start.dtype, is_block=True, shape=start.shape))

            # Emit 2-4 ops building on this chain variable
            ops_in_chain = rng.randint(2, 4)
            for _ in range(ops_in_chain):
                self._emit_one_op()

            # The last emitted variable is this chain's output
            last = self.symtab.last_var()
            if last is not None:
                chain_vars.append(last)

        # Merge all chain outputs into a single variable so the compiler
        # must keep every chain live until this point.
        if len(chain_vars) >= 2:
            merge_name = chain_vars[0].name
            out_dt = chain_vars[0].dtype
            for cv in chain_vars[1:]:
                new_name = self.symtab.fresh_name("v")
                self._triton_body.append(f"{new_name} = {merge_name} + {cv.name}")
                self._torch_body.append(f"{new_name} = {merge_name} + {cv.name}")
                out_dt = promote(out_dt, cv.dtype)
                self.symtab.add(TensorVar(new_name, out_dt, is_block=True, shape=chain_vars[0].shape))
                merge_name = new_name

        self._ops_used.append("parallel_chains")

    # ── Scalar condition helpers (for if/else and while) ─────────────────

    def _make_scalar_condition(self) -> tuple[str, str]:
        """Build a scalar boolean condition from a reduction of a block var.

        Returns ``(triton_cond_expr, torch_cond_expr)`` suitable for use
        in ``if`` / ``while`` headers.  The condition reduces a block
        tensor to a scalar via ``tl.sum`` / ``tl.max`` / ``tl.min`` and
        compares it against a constant threshold.
        """
        rng = self.rng
        cond_var = self.symtab.pick_random(rng, is_block=True, shape=())
        if cond_var is None:
            # Fallback: always-true to keep generation going
            return "True", "True"

        idx = rng.randint(0, len(_COND_REDUCTION_TRITON) - 1)
        triton_red = _COND_REDUCTION_TRITON[idx].format(cond_var.name)
        torch_red = _COND_REDUCTION_TORCH[idx].format(cond_var.name)

        cmp = rng.choice(_COND_COMPARISONS)
        threshold = rng.choice(_COND_THRESHOLDS)

        triton_cond = f"{triton_red} {cmp} {threshold}"
        torch_cond = f"{torch_red} {cmp} {threshold}"
        return triton_cond, torch_cond

    # ── If/else branch emission (divergent control flow) ─────────────────

    def _emit_if_else(self, indent: str = "") -> None:
        """Emit an ``if``/``else`` block where each branch performs
        different tensor operations, forcing the LLVM backend to handle
        CFG reconvergence and divergent register allocation.

        Pattern::

            _cond = tl.sum(v_M, axis=0) > 0.5
            if _cond:
                v_N = tl.exp(v_A)
            else:
                v_N = tl.sin(v_B)

        Both branches write to the **same output variable** so the
        post-branch symbol table stays consistent.
        """
        rng = self.rng
        triton_cond, torch_cond = self._make_scalar_condition()

        # Pre-allocate the output name that both branches must define.
        out_name = self.symtab.fresh_name("v")
        inner = indent + "    "

        # Snapshot before branching so both sides see the same operands.
        snap = self.symtab.snapshot()

        # ── IF header ────────────────────────────────────────────────
        self._triton_body.append(f"{indent}if {triton_cond}:")
        self._torch_body.append(f"{indent}if {torch_cond}:")

        # ── True branch ──────────────────────────────────────────────
        true_triton_start = len(self._triton_body)
        true_torch_start = len(self._torch_body)
        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent=inner)
        true_last = self.symtab.last_var()

        # Assign to unified output
        if true_last is not None:
            self._triton_body.append(f"{inner}{out_name} = {true_last.name}")
            self._torch_body.append(f"{inner}{out_name} = {true_last.name}")
            true_dtype = true_last.dtype
            true_shape = true_last.shape
        else:
            # Branch produced nothing — fallback to an existing var
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"{inner}{out_name} = {fb_name}")
            self._torch_body.append(f"{inner}{out_name} = {fb_name}")
            true_dtype = fb.dtype if fb else FLOAT32
            true_shape = fb.shape if fb else ()

        # ── Restore for false branch ─────────────────────────────────
        self.symtab.restore(snap)

        # ── ELSE header ──────────────────────────────────────────────
        self._triton_body.append(f"{indent}else:")
        self._torch_body.append(f"{indent}else:")

        # ── False branch ─────────────────────────────────────────────
        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent=inner)
        false_last = self.symtab.last_var()

        if false_last is not None:
            self._triton_body.append(f"{inner}{out_name} = {false_last.name}")
            self._torch_body.append(f"{inner}{out_name} = {false_last.name}")
            false_dtype = false_last.dtype
            false_shape = false_last.shape
        else:
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"{inner}{out_name} = {fb_name}")
            self._torch_body.append(f"{inner}{out_name} = {fb_name}")
            false_dtype = fb.dtype if fb else FLOAT32
            false_shape = fb.shape if fb else ()

        # ── Restore and register unified output ──────────────────────
        self.symtab.restore(snap)
        out_dt = promote(true_dtype, false_dtype)
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=true_shape))
        self._ops_used.append("if_else")

    # ── While-loop emission (data-dependent bounded iteration) ───────────

    def _emit_while_loop(self, indent: str = "") -> None:
        """Emit a data-dependent ``while``-loop with a mandatory counter
        guard to prevent infinite iteration.

        Pattern::

            v_N = <source>
            _while_i = 0
            while tl.max(v_N, axis=0) < 10.0 and _while_i < MAX_ITER:
                v_N = v_N <op> <operand>
                _while_i += 1

        The data-dependent condition (reduction comparison) forces the
        Triton → MLIR → LLVM pipeline to handle ``scf.while`` lowering,
        while the counter guard keeps execution bounded.
        """
        rng = self.rng
        src = self.symtab.pick_random(rng, is_block=True, shape=())
        operand = self.symtab.pick_random(rng, is_block=True, shape=())
        if src is None or operand is None:
            return

        acc_name = self.symtab.fresh_name("v")
        counter_name = f"_while_i_{self.symtab._counter}"
        accum_op = rng.choice(_LOOP_ACCUM_OPS)
        max_iter = self.while_max_iter

        # Build the data-dependent condition
        idx = rng.randint(0, len(_COND_REDUCTION_TRITON) - 1)
        triton_red = _COND_REDUCTION_TRITON[idx]
        torch_red = _COND_REDUCTION_TORCH[idx]
        cmp = rng.choice(_COND_COMPARISONS)
        threshold = rng.choice(["2.0", "5.0", "10.0", "20.0"])

        inner = indent + "    "

        # Initialise accumulator and counter
        self._triton_body.append(f"{indent}{acc_name} = {src.name}")
        self._torch_body.append(f"{indent}{acc_name} = {src.name}")
        self._triton_body.append(f"{indent}{counter_name} = 0")
        self._torch_body.append(f"{indent}{counter_name} = 0")

        # While header with data-dependent + counter condition
        triton_cond = f"{triton_red.format(acc_name)} {cmp} {threshold} and {counter_name} < {max_iter}"
        torch_cond = f"{torch_red.format(acc_name)} {cmp} {threshold} and {counter_name} < {max_iter}"
        self._triton_body.append(f"{indent}while {triton_cond}:")
        self._torch_body.append(f"{indent}while {torch_cond}:")

        # Loop body: accumulation + counter increment
        self._triton_body.append(
            f"{inner}{acc_name} = {acc_name} {accum_op} {operand.name}"
        )
        self._torch_body.append(
            f"{inner}{acc_name} = {acc_name} {accum_op} {operand.name}"
        )
        self._triton_body.append(f"{inner}{counter_name} += 1")
        self._torch_body.append(f"{inner}{counter_name} += 1")

        out_dt = promote(src.dtype, operand.dtype)
        self.symtab.add(TensorVar(acc_name, out_dt, is_block=True, shape=src.shape))
        self._ops_used.append("while_loop")

    # ── Nested control-flow emission ─────────────────────────────────────

    def _emit_nested_cf(self) -> None:
        """Emit nested control-flow according to ``self.nested_cf_pattern``.

        Supported patterns:

        * ``if_in_if``    – outer if/else where one branch contains an
                            inner if/else (depth-2 branching, stresses
                            reconvergence tracking).
        * ``if_in_while`` – while-loop whose body contains an if/else
                            (loop→branch nesting, stresses register
                            allocation across iterations with divergent
                            paths).
        * ``while_in_if`` – outer if/else where one branch contains a
                            while-loop (branch→loop nesting, stresses
                            shared-memory synchronisation).
        """
        pattern = self.nested_cf_pattern
        if pattern == "if_in_if":
            self._emit_nested_if_in_if()
        elif pattern == "if_in_while":
            self._emit_loop_with_branch()
        elif pattern == "while_in_if":
            self._emit_while_in_if()
        else:
            # Fallback: simple if/else
            self._emit_if_else()

    def _emit_nested_if_in_if(self) -> None:
        """Outer if/else where the true branch contains an inner if/else.

        Pattern::

            if <cond_outer>:
                if <cond_inner>:
                    v_N = <op_a>(v_X)
                else:
                    v_N = <op_b>(v_X)
            else:
                v_N = <op_c>(v_X)
        """
        rng = self.rng
        triton_cond_outer, torch_cond_outer = self._make_scalar_condition()
        out_name = self.symtab.fresh_name("v")
        snap = self.symtab.snapshot()

        # ── Outer IF ─────────────────────────────────────────────────
        self._triton_body.append(f"if {triton_cond_outer}:")
        self._torch_body.append(f"if {torch_cond_outer}:")

        # Inner if/else inside the true branch (indent = 4 spaces)
        triton_cond_inner, torch_cond_inner = self._make_scalar_condition()
        inner_snap = self.symtab.snapshot()

        self._triton_body.append(f"    if {triton_cond_inner}:")
        self._torch_body.append(f"    if {torch_cond_inner}:")

        # Inner true branch (indent = 8 spaces)
        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent="        ")
        inner_true_last = self.symtab.last_var()
        if inner_true_last is not None:
            self._triton_body.append(f"        {out_name} = {inner_true_last.name}")
            self._torch_body.append(f"        {out_name} = {inner_true_last.name}")
            inner_true_dtype = inner_true_last.dtype
        else:
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"        {out_name} = {fb_name}")
            self._torch_body.append(f"        {out_name} = {fb_name}")
            inner_true_dtype = fb.dtype if fb else FLOAT32

        # Restore for inner else
        self.symtab.restore(inner_snap)

        self._triton_body.append("    else:")
        self._torch_body.append("    else:")

        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent="        ")
        inner_false_last = self.symtab.last_var()
        if inner_false_last is not None:
            self._triton_body.append(f"        {out_name} = {inner_false_last.name}")
            self._torch_body.append(f"        {out_name} = {inner_false_last.name}")
            inner_false_dtype = inner_false_last.dtype
        else:
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"        {out_name} = {fb_name}")
            self._torch_body.append(f"        {out_name} = {fb_name}")
            inner_false_dtype = fb.dtype if fb else FLOAT32

        # ── Outer ELSE ───────────────────────────────────────────────
        self.symtab.restore(snap)

        self._triton_body.append("else:")
        self._torch_body.append("else:")

        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent="    ")
        outer_false_last = self.symtab.last_var()
        if outer_false_last is not None:
            self._triton_body.append(f"    {out_name} = {outer_false_last.name}")
            self._torch_body.append(f"    {out_name} = {outer_false_last.name}")
            outer_false_dtype = outer_false_last.dtype
        else:
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"    {out_name} = {fb_name}")
            self._torch_body.append(f"    {out_name} = {fb_name}")
            outer_false_dtype = fb.dtype if fb else FLOAT32

        # ── Register unified output ──────────────────────────────────
        self.symtab.restore(snap)
        out_dt = promote(promote(inner_true_dtype, inner_false_dtype), outer_false_dtype)
        src_for_shape = self.symtab.pick_random(rng, is_block=True, shape=())
        shape = src_for_shape.shape if src_for_shape else ()
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=shape))
        self._ops_used.append("nested_if_in_if")

    def _emit_loop_with_branch(self) -> None:
        """Emit a while-loop whose body contains an if/else branch.

        Pattern::

            v_N = <source>
            _while_i = 0
            while <data_cond> and _while_i < MAX_ITER:
                if <inner_cond>:
                    v_N = v_N + v_op1
                else:
                    v_N = v_N * v_op2
                _while_i += 1

        This loop→branch nesting forces the compiler to handle
        reconvergence inside a loop body, stressing the ``scf.while``
        lowering combined with ``scf.if`` nesting.
        """
        rng = self.rng
        src = self.symtab.pick_random(rng, is_block=True, shape=())
        op1 = self.symtab.pick_random(rng, is_block=True, shape=())
        op2 = self.symtab.pick_random(rng, is_block=True, shape=())
        if src is None or op1 is None or op2 is None:
            return

        acc_name = self.symtab.fresh_name("v")
        counter_name = f"_while_i_{self.symtab._counter}"
        max_iter = self.while_max_iter
        accum_op_true = rng.choice(_LOOP_ACCUM_OPS)
        accum_op_false = rng.choice(_LOOP_ACCUM_OPS)

        # Data-dependent loop condition
        idx = rng.randint(0, len(_COND_REDUCTION_TRITON) - 1)
        triton_red = _COND_REDUCTION_TRITON[idx]
        torch_red = _COND_REDUCTION_TORCH[idx]
        cmp = rng.choice(_COND_COMPARISONS)
        threshold = rng.choice(["2.0", "5.0", "10.0", "20.0"])

        # Inner branch condition (different reduction)
        idx2 = rng.randint(0, len(_COND_REDUCTION_TRITON) - 1)
        triton_inner_red = _COND_REDUCTION_TRITON[idx2]
        torch_inner_red = _COND_REDUCTION_TORCH[idx2]
        inner_cmp = rng.choice(_COND_COMPARISONS)
        inner_threshold = rng.choice(_COND_THRESHOLDS)

        # Initialise
        self._triton_body.append(f"{acc_name} = {src.name}")
        self._torch_body.append(f"{acc_name} = {src.name}")
        self._triton_body.append(f"{counter_name} = 0")
        self._torch_body.append(f"{counter_name} = 0")

        # While header
        triton_cond = f"{triton_red.format(acc_name)} {cmp} {threshold} and {counter_name} < {max_iter}"
        torch_cond = f"{torch_red.format(acc_name)} {cmp} {threshold} and {counter_name} < {max_iter}"
        self._triton_body.append(f"while {triton_cond}:")
        self._torch_body.append(f"while {torch_cond}:")

        # Inner if/else
        triton_inner_cond = f"{triton_inner_red.format(acc_name)} {inner_cmp} {inner_threshold}"
        torch_inner_cond = f"{torch_inner_red.format(acc_name)} {inner_cmp} {inner_threshold}"
        self._triton_body.append(f"    if {triton_inner_cond}:")
        self._torch_body.append(f"    if {torch_inner_cond}:")

        self._triton_body.append(
            f"        {acc_name} = {acc_name} {accum_op_true} {op1.name}"
        )
        self._torch_body.append(
            f"        {acc_name} = {acc_name} {accum_op_true} {op1.name}"
        )

        self._triton_body.append("    else:")
        self._torch_body.append("    else:")

        self._triton_body.append(
            f"        {acc_name} = {acc_name} {accum_op_false} {op2.name}"
        )
        self._torch_body.append(
            f"        {acc_name} = {acc_name} {accum_op_false} {op2.name}"
        )

        # Counter increment
        self._triton_body.append(f"    {counter_name} += 1")
        self._torch_body.append(f"    {counter_name} += 1")

        out_dt = promote(promote(src.dtype, op1.dtype), op2.dtype)
        self.symtab.add(TensorVar(acc_name, out_dt, is_block=True, shape=src.shape))
        self._ops_used.append("loop_with_branch")

    def _emit_while_in_if(self) -> None:
        """Emit an if/else where the true branch contains a while-loop.

        Pattern::

            if <cond>:
                v_N = <source>
                _while_i = 0
                while <data_cond> and _while_i < MAX_ITER:
                    v_N = v_N <op> <operand>
                    _while_i += 1
            else:
                v_N = <simple_op>(v_X)
        """
        rng = self.rng
        triton_cond, torch_cond = self._make_scalar_condition()
        out_name = self.symtab.fresh_name("v")
        snap = self.symtab.snapshot()

        # ── IF header ────────────────────────────────────────────────
        self._triton_body.append(f"if {triton_cond}:")
        self._torch_body.append(f"if {torch_cond}:")

        # True branch: while-loop (indented by 4)
        src = self.symtab.pick_random(rng, is_block=True, shape=())
        operand = self.symtab.pick_random(rng, is_block=True, shape=())
        if src is None or operand is None:
            # Fallback: emit a simple op
            self._emit_one_op(indent="    ")
            true_last = self.symtab.last_var()
            if true_last:
                self._triton_body.append(f"    {out_name} = {true_last.name}")
                self._torch_body.append(f"    {out_name} = {true_last.name}")
                true_dtype = true_last.dtype
                true_shape = true_last.shape
            else:
                self._triton_body.append(f"    {out_name} = 0.0")
                self._torch_body.append(f"    {out_name} = 0.0")
                true_dtype = FLOAT32
                true_shape = ()
        else:
            counter_name = f"_while_i_{self.symtab._counter}"
            accum_op = rng.choice(_LOOP_ACCUM_OPS)
            max_iter = self.while_max_iter

            idx = rng.randint(0, len(_COND_REDUCTION_TRITON) - 1)
            triton_red = _COND_REDUCTION_TRITON[idx]
            torch_red = _COND_REDUCTION_TORCH[idx]
            loop_cmp = rng.choice(_COND_COMPARISONS)
            loop_threshold = rng.choice(["2.0", "5.0", "10.0", "20.0"])

            self._triton_body.append(f"    {out_name} = {src.name}")
            self._torch_body.append(f"    {out_name} = {src.name}")
            self._triton_body.append(f"    {counter_name} = 0")
            self._torch_body.append(f"    {counter_name} = 0")

            triton_loop_cond = (
                f"{triton_red.format(out_name)} {loop_cmp} {loop_threshold}"
                f" and {counter_name} < {max_iter}"
            )
            torch_loop_cond = (
                f"{torch_red.format(out_name)} {loop_cmp} {loop_threshold}"
                f" and {counter_name} < {max_iter}"
            )
            self._triton_body.append(f"    while {triton_loop_cond}:")
            self._torch_body.append(f"    while {torch_loop_cond}:")

            self._triton_body.append(
                f"        {out_name} = {out_name} {accum_op} {operand.name}"
            )
            self._torch_body.append(
                f"        {out_name} = {out_name} {accum_op} {operand.name}"
            )
            self._triton_body.append(f"        {counter_name} += 1")
            self._torch_body.append(f"        {counter_name} += 1")

            true_dtype = promote(src.dtype, operand.dtype)
            true_shape = src.shape

        # ── ELSE branch ──────────────────────────────────────────────
        self.symtab.restore(snap)

        self._triton_body.append("else:")
        self._torch_body.append("else:")

        for _ in range(self.if_else_branch_ops):
            self._emit_one_op(indent="    ")
        false_last = self.symtab.last_var()
        if false_last is not None:
            self._triton_body.append(f"    {out_name} = {false_last.name}")
            self._torch_body.append(f"    {out_name} = {false_last.name}")
            false_dtype = false_last.dtype
            false_shape = false_last.shape
        else:
            fb = self.symtab.pick_random(rng)
            fb_name = fb.name if fb else "0.0"
            self._triton_body.append(f"    {out_name} = {fb_name}")
            self._torch_body.append(f"    {out_name} = {fb_name}")
            false_dtype = fb.dtype if fb else FLOAT32
            false_shape = fb.shape if fb else ()

        # ── Register unified output ──────────────────────────────────
        self.symtab.restore(snap)
        out_dt = promote(true_dtype, false_dtype)
        self.symtab.add(TensorVar(out_name, out_dt, is_block=True, shape=true_shape))
        self._ops_used.append("while_in_if")

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
        if self.use_block_ptr:
            return self._assemble_block_ptr_triton_source()
        if self.use_gather:
            return self._assemble_gather_triton_source()

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

    def _assemble_block_ptr_triton_source(self) -> str:
        """Assemble a kernel using ``tl.make_block_ptr`` for loads/stores.

        The structured-memory path (block pointers) exercises Triton's
        memory hierarchy optimisations, including coalescing analysis,
        swizzling, and ``tl.advance``-based pointer arithmetic.
        """
        seed = self.seed
        out_var = self._output_var.name if self._output_var else "v_0"

        # Build parameter list: raw base pointers + n_elements + BLOCK_SIZE
        ptr_args = ", ".join(f"in_ptr{i}" for i in range(self.num_inputs))

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
            "    pid = tl.program_id(axis=0)",
        ]

        # Create block pointers for each input
        for i in range(self.num_inputs):
            lines.extend([
                f"    in_blk_ptr{i} = tl.make_block_ptr(",
                f"        base=in_ptr{i},",
                "        shape=(n_elements,),",
                "        strides=(1,),",
                "        offsets=(pid * BLOCK_SIZE,),",
                "        block_shape=(BLOCK_SIZE,),",
                "        order=(0,),",
                "    )",
            ])

        # Create block pointer for output
        lines.extend([
            "    out_blk_ptr = tl.make_block_ptr(",
            "        base=out_ptr,",
            "        shape=(n_elements,),",
            "        strides=(1,),",
            "        offsets=(pid * BLOCK_SIZE,),",
            "        block_shape=(BLOCK_SIZE,),",
            "        order=(0,),",
            "    )",
        ])

        # Body (loads via block_ptr + computational ops)
        for line in self._triton_body:
            lines.append(f"    {line}")

        # Store via block pointer
        lines.append(f"    tl.store(out_blk_ptr, {out_var})")

        return "\n".join(lines) + "\n"

    def _assemble_gather_triton_source(self) -> str:
        """Assemble a kernel with indirect (gather) memory accesses.

        Input 0 is an int32 index tensor.  Data inputs use
        ``tl.load(data_ptr + idx)`` where ``idx`` comes from loading
        the index tensor, exercising unstructured gather instructions.
        """
        seed = self.seed
        out_var = self._output_var.name if self._output_var else "v_0"

        # Build parameter list: idx_ptr, in_ptr1, in_ptr2, ..., out_ptr
        data_ptrs = ", ".join(f"in_ptr{i}" for i in range(1, self.num_inputs))
        all_ptrs = f"idx_ptr, {data_ptrs}" if data_ptrs else "idx_ptr"

        lines: list[str] = [
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def triton_kernel_seed_{seed}(",
            f"    {all_ptrs}, out_ptr,",
            "    n_elements,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    pid = tl.program_id(axis=0)",
            "    block_start = pid * BLOCK_SIZE",
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)",
        ]

        if self.use_mask:
            lines.append("    mask = offsets < n_elements")

        # Body (includes gather loads + computational ops)
        for line in self._triton_body:
            lines.append(f"    {line}")

        # Store epilogue
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
        if self.use_multi_dot:
            return self._assemble_multi_dot_triton_source()

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

    def _assemble_multi_dot_triton_source(self) -> str:
        """Assemble a chained-dot kernel: dot(A,B) → trans → dot(T,C)."""
        seed = self.seed
        acc1 = self._dot_acc_name
        acc2 = self._multi_dot_acc2_name
        out_var = self._output_var.name if self._output_var else acc2

        lines: list[str] = [
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def triton_kernel_seed_{seed}(",
            "    a_ptr, b_ptr, c_ptr, out_ptr,",
            "    M, N, K, P,",
            "    stride_am, stride_ak,",
            "    stride_bk, stride_bn,",
            "    stride_cn2, stride_cp,",
            "    stride_om, stride_op,",
            "    BLOCK_M: tl.constexpr,",
            "    BLOCK_N: tl.constexpr,",
            "    BLOCK_K: tl.constexpr,",
            "    BLOCK_P: tl.constexpr,",
            "):",
            "    pid_m = tl.program_id(0)",
            "    pid_n = tl.program_id(1)",
            "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            # First dot: A @ B → acc1 (BLOCK_M, BLOCK_N)
            f"    {acc1} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
            "    for k_start in range(0, K, BLOCK_K):",
            "        offs_k = k_start + tl.arange(0, BLOCK_K)",
            "        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)",
            "        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)",
            f"        {acc1} += tl.dot(a, b)",
        ]

        # Body includes: trans(acc1), then second dot is inline
        # The body lines handle: trans_name = tl.trans(acc1), acc2 = ...
        # But we also need the second K-loop for dot(trans, C).
        # Since multi-dot body already emitted trans into _triton_body,
        # we emit the second loop here with acc2 as accumulator.

        # Emit the trans line from body (first line is the trans)
        if self._triton_body:
            lines.append(f"    {self._triton_body[0]}")  # trans line

        # Second dot: trans(acc1) @ C → acc2
        lines.extend([
            f"    offs_p = pid_n * BLOCK_P + tl.arange(0, BLOCK_P)",
            f"    {acc2} = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)",
            "    for n_start in range(0, N, BLOCK_N):",
            "        offs_n2 = n_start + tl.arange(0, BLOCK_N)",
            "        c_tile = tl.load(c_ptr + offs_n2[:, None] * stride_cn2 + offs_p[None, :] * stride_cp)",
        ])
        # The trans result has shape (BLOCK_N, BLOCK_M); since M==N, this is fine
        # We use it as the left operand to dot
        trans_name = self._triton_body[0].split("=")[0].strip() if self._triton_body else acc1
        lines.append(f"        {acc2} += tl.dot({trans_name}.to(tl.float16), c_tile)")

        # Remaining body lines (post-ops after the trans and acc2 registration)
        for line in self._triton_body[1:]:
            lines.append(f"    {line}")

        # Store
        lines.extend([
            "    c_out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_op",
            f"    tl.store(c_out, {out_var})",
        ])

        return "\n".join(lines) + "\n"

    def _assemble_torch_ref_source(self) -> str:
        if self.use_dot:
            return self._assemble_dot_torch_ref_source()

        seed = self.seed
        out_var = self._output_var.name if self._output_var else "v_0"

        # Build parameter list matching the kernel convention
        if self.use_gather:
            # First arg is the index tensor, then data tensors
            args = ["x_idx"] + [f"x{i}" for i in range(1, self.num_inputs)]
            input_args = ", ".join(args)
        else:
            input_args = ", ".join(f"x{i}" for i in range(self.num_inputs))

        lines: list[str] = [
            "import torch",
            "",
            "",
            f"def torch_ref_seed_{seed}({input_args}):",
        ]

        # For gather mode, the torch ref needs to slice data tensors
        # using the index.  The _torch_body already uses x_idx and x1[idx_0].
        for line in self._torch_body:
            lines.append(f"    {line}")

        lines.append(f"    return {out_var}")

        return "\n".join(lines) + "\n"

    def _assemble_dot_torch_ref_source(self) -> str:
        """Assemble the PyTorch reference for a dot kernel."""
        if self.use_multi_dot:
            return self._assemble_multi_dot_torch_ref_source()

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

        # When layout ops are used, the body references M, N, BLOCK_M,
        # BLOCK_N via reshape / expand calls.  Derive them from the
        # accumulator's actual shape so the reference is self-contained.
        if self.dot_use_layout_stress:
            lines.append(f"    M, N = {acc}.shape")
            lines.append(f"    BLOCK_M, BLOCK_N = M, N")

        for line in self._torch_body:
            lines.append(f"    {line}")

        lines.append(f"    return {out_var}")

        return "\n".join(lines) + "\n"

    def _assemble_multi_dot_torch_ref_source(self) -> str:
        """Assemble the PyTorch reference for a chained-dot kernel."""
        seed = self.seed
        acc1 = self._dot_acc_name
        acc2 = self._multi_dot_acc2_name
        out_var = self._output_var.name if self._output_var else acc2

        lines: list[str] = [
            "import torch",
            "",
            "",
            f"def torch_ref_seed_{seed}(a, b, c):",
            f"    {acc1} = torch.matmul(a.float(), b.float())",
        ]

        # First torch_body line is the trans
        if self._torch_body:
            lines.append(f"    {self._torch_body[0]}")

        # Second matmul: trans(acc1) @ c
        trans_name = self._torch_body[0].split("=")[0].strip() if self._torch_body else acc1
        lines.append(f"    {acc2} = torch.matmul({trans_name}.half().float(), c.float())")

        # Derive shape vars for layout ops
        lines.append(f"    M, N = {acc2}.shape")
        lines.append(f"    BLOCK_M, BLOCK_N = M, N")

        # Remaining body lines (skip first which was trans)
        for line in self._torch_body[1:]:
            lines.append(f"    {line}")

        lines.append(f"    return {out_var}")

        return "\n".join(lines) + "\n"

    # ================================================================== #
    #  Metadata                                                            #
    # ================================================================== #

    def _build_metadata(self) -> dict:
        meta = {
            "generator_version": "0.5.0",
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
            # ── Divergent control flow ───────────────────────────────
            "has_if_else": self.insert_if_else,
            "has_while_loop": self.insert_while_loop,
            "while_max_iter": self.while_max_iter if self.insert_while_loop or self.insert_nested_cf else 0,
            "has_nested_cf": self.insert_nested_cf,
            "nested_cf_pattern": self.nested_cf_pattern,
            # ── Pointer math modes ───────────────────────────────────
            "use_block_ptr": self.use_block_ptr,
            "use_gather": self.use_gather,
            "gather_data_inputs": self.gather_data_inputs if self.use_gather else 0,
            # ── Register-pressure stress ─────────────────────────────
            "use_reg_pressure": self.use_reg_pressure,
        }
        if self.use_reg_pressure:
            meta["reg_pressure_num_warps"] = self.reg_pressure_num_warps
            meta["reg_pressure_num_stages"] = self.reg_pressure_num_stages
            meta["reg_pressure_parallel_chains"] = self.reg_pressure_parallel_chains
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
            # ── Layout-conversion stress ─────────────────────────────
            meta["dot_layout_stress"] = self.dot_use_layout_stress
            meta["dot_layout_ops"] = self.dot_layout_ops
            meta["dot_square_blocks"] = self.dot_square_blocks
            layout_ops = [
                op for op in self._ops_used
                if op in ("trans", "reshape_flatten", "reshape_unflatten",
                          "expand_broadcast", "trans_fixup", "reshape_fixup")
            ]
            meta["layout_ops_used"] = layout_ops
            # ── Multi-dot ────────────────────────────────────────────
            meta["multi_dot"] = self.use_multi_dot
            if self.use_multi_dot:
                meta["multi_dot_P"] = self.multi_dot_P
                meta["multi_dot_BLOCK_P"] = self.multi_dot_BLOCK_P
                meta["num_inputs"] = 3
                meta["input_shapes"] = [
                    [self.dot_M, self.dot_K],
                    [self.dot_K, self.dot_N],
                    [self.dot_N, self.multi_dot_P],
                ]
                meta["output_shape"] = [self.dot_M, self.multi_dot_P]
        return meta
