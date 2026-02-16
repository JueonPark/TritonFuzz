"""Triton and PyTorch data-type definitions, registry, and promotion rules.

Triton supports a rich type system (``int1`` through ``int64``, ``float16``,
``bfloat16``, ``float32``, ``float64``).  This module provides a unified
``DType`` descriptor that carries both the Triton and PyTorch spelling of
each type, together with helper functions for random selection and implicit
type promotion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# ── Core data class ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DType:
    """A data type with Triton, PyTorch, and short-hand representations."""

    triton: str       # e.g. "tl.float32"
    torch: str        # e.g. "torch.float32"
    short: str        # e.g. "fp32"  (used in variable naming / metadata)
    is_float: bool
    bits: int

    @property
    def is_int(self) -> bool:
        return not self.is_float


# ── Concrete dtype instances ─────────────────────────────────────────────────

FLOAT16  = DType("tl.float16",  "torch.float16",  "fp16",  True,  16)
BFLOAT16 = DType("tl.bfloat16", "torch.bfloat16", "bf16",  True,  16)
FLOAT32  = DType("tl.float32",  "torch.float32",  "fp32",  True,  32)
FLOAT64  = DType("tl.float64",  "torch.float64",  "fp64",  True,  64)

INT1  = DType("tl.int1",  "torch.bool",  "i1",  False,  1)
INT8  = DType("tl.int8",  "torch.int8",  "i8",  False,  8)
INT16 = DType("tl.int16", "torch.int16", "i16", False, 16)
INT32 = DType("tl.int32", "torch.int32", "i32", False, 32)
INT64 = DType("tl.int64", "torch.int64", "i64", False, 64)

# ── Grouped collections ──────────────────────────────────────────────────────

ALL_DTYPES: list[DType] = [
    FLOAT16, BFLOAT16, FLOAT32, FLOAT64,
    INT8, INT16, INT32, INT64,
]

FLOAT_DTYPES: list[DType] = [d for d in ALL_DTYPES if d.is_float]
INT_DTYPES: list[DType]   = [d for d in ALL_DTYPES if d.is_int and d.bits > 1]

# Commonly chosen for generated kernel inputs (biased toward "problematic" types).
INPUT_DTYPES: list[DType]       = [FLOAT16, BFLOAT16, FLOAT32, INT32]
_INPUT_DTYPE_WEIGHTS: list[float] = [2.0,    2.0,      3.0,     1.0]

# Types we cast *to* during type-stress tests.
CAST_TARGET_DTYPES: list[DType] = [FLOAT16, BFLOAT16, FLOAT32, INT32]


# ── Selection helpers ────────────────────────────────────────────────────────


def pick_input_dtype(rng: random.Random) -> DType:
    """Randomly select a dtype for a kernel input tensor."""
    return rng.choices(INPUT_DTYPES, weights=_INPUT_DTYPE_WEIGHTS, k=1)[0]


def pick_float_dtype(rng: random.Random) -> DType:
    """Pick a float dtype (for ops that require floating-point input)."""
    return rng.choice([FLOAT16, BFLOAT16, FLOAT32])


# ── Promotion logic ─────────────────────────────────────────────────────────


def promote(a: DType, b: DType) -> DType:
    """Return the 'wider' type, following Triton / PyTorch implicit rules.

    * Float always wins over int.
    * Among same kind, larger ``bits`` wins.
    """
    if a.is_float and not b.is_float:
        return a
    if b.is_float and not a.is_float:
        return b
    # Same kind → wider bits wins
    return a if a.bits >= b.bits else b
