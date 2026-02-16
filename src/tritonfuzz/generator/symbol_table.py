"""Symbol table for tracking live tensor variables during kernel generation.

During code synthesis the ``SymbolTable`` keeps track of every named
tensor that has been loaded or computed, along with its dtype and shape
class (``block``-shaped vs. scalar).  The body builder queries the table to
find suitable operands for the next operation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from tritonfuzz.generator.types import DType


@dataclass
class TensorVar:
    """A named tensor variable tracked during generation."""

    name: str
    dtype: DType
    is_block: bool = True  # True → BLOCK_SIZE-shaped; False → scalar


class SymbolTable:
    """Registry of live variables available at any point during synthesis."""

    def __init__(self) -> None:
        self._vars: dict[str, TensorVar] = {}
        self._counter: int = 0
        self._order: list[str] = []      # insertion order

    # ── Mutation ─────────────────────────────────────────────────────────

    def add(self, var: TensorVar) -> None:
        """Insert (or overwrite) a variable."""
        self._vars[var.name] = var
        if var.name not in self._order:
            self._order.append(var.name)

    def fresh_name(self, prefix: str = "v") -> str:
        """Return a monotonically-increasing unique name."""
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    # ── Queries ──────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[TensorVar]:
        return self._vars.get(name)

    def all_vars(self) -> list[TensorVar]:
        """All variables in insertion order."""
        return [self._vars[n] for n in self._order if n in self._vars]

    def block_vars(self, *, require_float: bool = False) -> list[TensorVar]:
        """Return block-shaped variables, optionally filtered to floats."""
        result = [v for v in self.all_vars() if v.is_block]
        if require_float:
            result = [v for v in result if v.dtype.is_float]
        return result

    def pick_random(
        self,
        rng: random.Random,
        *,
        is_block: bool = True,
        require_float: bool = False,
    ) -> Optional[TensorVar]:
        """Pick a random variable matching the filters, or ``None``."""
        candidates = [v for v in self.all_vars() if v.is_block == is_block]
        if require_float:
            candidates = [v for v in candidates if v.dtype.is_float]
        return rng.choice(candidates) if candidates else None

    def pick_two(
        self,
        rng: random.Random,
        *,
        is_block: bool = True,
        prefer_same_dtype: bool = False,
    ) -> tuple[Optional[TensorVar], Optional[TensorVar]]:
        """Pick two (possibly identical) variables for binary ops."""
        a = self.pick_random(rng, is_block=is_block)
        if a is None:
            return None, None
        if prefer_same_dtype:
            same = [v for v in self.all_vars() if v.is_block == is_block and v.dtype == a.dtype]
            b = rng.choice(same) if same else self.pick_random(rng, is_block=is_block)
        else:
            b = self.pick_random(rng, is_block=is_block)
        return a, b

    def last_var(self) -> Optional[TensorVar]:
        """Return the most recently inserted variable, or ``None``."""
        if not self._order:
            return None
        return self._vars.get(self._order[-1])

    def __len__(self) -> int:
        return len(self._vars)
