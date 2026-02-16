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

    # Future stubs
    REDUCTION    = "reduction"
    DOT          = "dot"
    ATOMIC       = "atomic"
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

# ── Convenience aggregates ───────────────────────────────────────────────────

ALL_TEMPLATE_OPS: list[OpTemplate] = UNARY_OPS + BINARY_OPS + LOGIC_BINARY_OPS

# ── Category weights (how often each category is chosen) ─────────────────────

CATEGORY_WEIGHTS: dict[OpCategory, float] = {
    OpCategory.ELEMENTWISE_UNARY:  3.0,
    OpCategory.ELEMENTWISE_BINARY: 4.0,
    OpCategory.LOGIC:              2.0,
    OpCategory.TYPE_CAST:          1.5,
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
