"""Operation templates and registry for kernel body synthesis.

Each ``OpTemplate`` describes a single Triton language operation together with
its PyTorch equivalent.  The body builder picks from these templates —
weighted toward operations known to stress the compiler — to construct the
computational DAG inside a kernel.

Operator categories (from the design document):

====================  ========================================================
Category              Fuzzing strategy
====================  ========================================================
Element-wise unary    Chain deeply to exhaust registers; mix dtypes.
Element-wise binary   Same; exercise implicit type promotion.
Logic                 Complex boolean conditions for ``where``.
Type cast             Explicitly change precision to stress cast lowering.
Reduction             (Future) ``sum``, ``max``, ``min`` across ``axis=0``.
Linear Algebra        (Future) ``dot`` with MxK, KxN inputs.
Atomics               (Future) ``atomic_add``, ``atomic_cas``, ``atomic_max``.
Pointer Math          (Future) ``make_block_ptr``, ``advance``.
====================  ========================================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


# ── Category enum ────────────────────────────────────────────────────────────


class OpCategory(str, Enum):
    ELEMENTWISE_UNARY  = "elementwise_unary"
    ELEMENTWISE_BINARY = "elementwise_binary"
    LOGIC              = "logic"
    TYPE_CAST          = "type_cast"
    ATOMIC             = "atomic"
    DOT                = "dot"

    REDUCTION    = "reduction"

    # Future stubs
    POINTER_MATH = "pointer_math"


# ── Template dataclass ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpTemplate:
    """Describes one operation that can appear in the kernel body.

    ``triton_fmt`` / ``torch_fmt`` are Python *format strings* whose
    positional placeholders ``{0}``, ``{1}``, … are filled with variable
    names by the builder.
    """

    name: str
    category: OpCategory
    arity: int                 # number of tensor inputs (1 or 2)
    triton_fmt: str            # e.g. "tl.sin({0})"
    torch_fmt: str             # e.g. "torch.sin({0})"
    output_dtype_rule: str     # "same" | "promote" | "float32" | "bool"
    weight: float = 1.0        # sampling probability weight
    requires_float: bool = False


# ── Element-wise Unary operations ────────────────────────────────────────────

UNARY_OPS: list[OpTemplate] = [
    OpTemplate("exp",  OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.exp({0})",  "torch.exp({0})",  "same", 2.0, True),
    OpTemplate("sin",  OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.sin({0})",  "torch.sin({0})",  "same", 2.0, True),
    OpTemplate("cos",  OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.cos({0})",  "torch.cos({0})",  "same", 2.0, True),
    OpTemplate("log",  OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.log({0})",  "torch.log({0})",  "same", 1.5, True),
    OpTemplate("abs",  OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.abs({0})",  "torch.abs({0})",  "same", 1.0, False),
    OpTemplate("sqrt", OpCategory.ELEMENTWISE_UNARY, 1,
               "tl.sqrt({0})", "torch.sqrt({0})", "same", 1.5, True),
    OpTemplate("neg",  OpCategory.ELEMENTWISE_UNARY, 1,
               "-{0}",         "-{0}",            "same", 1.0, False),
]

# ── Element-wise Binary operations ───────────────────────────────────────────

BINARY_OPS: list[OpTemplate] = [
    OpTemplate("add", OpCategory.ELEMENTWISE_BINARY, 2,
               "{0} + {1}", "{0} + {1}", "promote", 3.0),
    OpTemplate("sub", OpCategory.ELEMENTWISE_BINARY, 2,
               "{0} - {1}", "{0} - {1}", "promote", 2.0),
    OpTemplate("mul", OpCategory.ELEMENTWISE_BINARY, 2,
               "{0} * {1}", "{0} * {1}", "promote", 3.0),
    OpTemplate("div", OpCategory.ELEMENTWISE_BINARY, 2,
               "{0} / {1}", "{0} / {1}", "promote", 1.5),
]

# ── Logic operations (template-based; ``where`` is handled specially) ────────

LOGIC_BINARY_OPS: list[OpTemplate] = [
    OpTemplate("minimum", OpCategory.LOGIC, 2,
               "tl.minimum({0}, {1})", "torch.minimum({0}, {1})", "promote", 2.0),
    OpTemplate("maximum", OpCategory.LOGIC, 2,
               "tl.maximum({0}, {1})", "torch.maximum({0}, {1})", "promote", 2.0),
]


# ── Atomic store operations (replace epilogue ``tl.store``) ──────────────────


@dataclass(frozen=True)
class AtomicOpTemplate:
    """Describes an atomic operation that replaces the store epilogue.

    Unlike regular ``OpTemplate``s which produce body statements, atomics
    change the *store* at the end of the kernel.  The ``output_init``
    field tells the runtime how to initialise the output tensor so that
    the atomic is an identity w.r.t. the first write (non-overlapping
    pattern).
    """

    name: str                # e.g. "atomic_add"
    triton_fn: str           # e.g. "tl.atomic_add"
    output_init: str         # "zeros" | "neg_inf" | "pos_inf" | "empty"
    weight: float = 1.0


ATOMIC_OPS: list[AtomicOpTemplate] = [
    AtomicOpTemplate("atomic_add",  "tl.atomic_add",  "zeros",   3.0),
    AtomicOpTemplate("atomic_max",  "tl.atomic_max",  "neg_inf", 2.0),
    AtomicOpTemplate("atomic_min",  "tl.atomic_min",  "pos_inf", 2.0),
    AtomicOpTemplate("atomic_xchg", "tl.atomic_xchg", "empty",   1.0),
]


def pick_atomic_op(rng: random.Random) -> AtomicOpTemplate:
    """Weighted random selection of an atomic store operation."""
    weights = [op.weight for op in ATOMIC_OPS]
    return rng.choices(ATOMIC_OPS, weights=weights, k=1)[0]


# ── Reduction operations (tl.sum, tl.max, tl.min) ───────────────────────────


@dataclass(frozen=True)
class ReductionOpTemplate:
    """Describes a reduction operation that collapses a block to a scalar.

    The builder emits the reduction then immediately combines the scalar
    result with a block-shaped variable (broadcasting), so the final
    registered variable remains block-shaped and can be stored normally.
    """

    name: str
    triton_fmt: str   # e.g. "tl.sum({0}, axis=0)"
    torch_fmt: str    # e.g. "torch.sum({0})"
    weight: float = 1.0
    requires_float: bool = False


REDUCTION_OPS: list[ReductionOpTemplate] = [
    ReductionOpTemplate("reduce_sum", "tl.sum({0}, axis=0)",  "torch.sum({0})",  3.0, False),
    ReductionOpTemplate("reduce_max", "tl.max({0}, axis=0)",  "torch.max({0})",  2.0, False),
    ReductionOpTemplate("reduce_min", "tl.min({0}, axis=0)",  "torch.min({0})",  2.0, False),
]


def pick_reduction_op(rng: random.Random) -> ReductionOpTemplate:
    """Weighted random selection of a reduction operation."""
    weights = [op.weight for op in REDUCTION_OPS]
    return rng.choices(REDUCTION_OPS, weights=weights, k=1)[0]

# ── Convenience aggregates ───────────────────────────────────────────────────

ALL_TEMPLATE_OPS: list[OpTemplate] = UNARY_OPS + BINARY_OPS + LOGIC_BINARY_OPS

# ── Category weights (how often each category is chosen) ─────────────────────

CATEGORY_WEIGHTS: dict[OpCategory, float] = {
    OpCategory.ELEMENTWISE_UNARY:  3.0,
    OpCategory.ELEMENTWISE_BINARY: 4.0,
    OpCategory.LOGIC:              2.0,
    OpCategory.TYPE_CAST:          1.5,
    OpCategory.REDUCTION:          1.5,
}

# ── Comparison primitives for ``tl.where`` condition expressions ─────────────

COMPARISON_OPS: list[str]        = [">", "<", ">=", "<="]
COMPARISON_THRESHOLDS: list[str] = ["0", "0.0", "0.5", "-0.5", "1.0"]


# ── Selection helpers ────────────────────────────────────────────────────────


def pick_category(rng: random.Random) -> OpCategory:
    """Weighted random category selection."""
    cats = list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in cats]
    return rng.choices(cats, weights=weights, k=1)[0]


def pick_op_from_category(rng: random.Random, category: OpCategory) -> OpTemplate | None:
    """Pick a random operation from *category* (weighted)."""
    if category == OpCategory.ELEMENTWISE_UNARY:
        pool = UNARY_OPS
    elif category == OpCategory.ELEMENTWISE_BINARY:
        pool = BINARY_OPS
    elif category == OpCategory.LOGIC:
        pool = LOGIC_BINARY_OPS
    else:
        return None  # TYPE_CAST / future categories handled by caller

    weights = [op.weight for op in pool]
    return rng.choices(pool, weights=weights, k=1)[0]
