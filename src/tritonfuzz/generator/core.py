"""Generator core – public ``Generator`` class and ``GeneratedKernel`` container.

This module is the main entry-point for the generation subsystem.  The
``Generator.generate(seed)`` method is called by the fuzzer loop; it
delegates to :class:`~tritonfuzz.generator.builder.KernelBuilder` for the
heavy lifting.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import Optional

from tritonfuzz.generator.builder import KernelBuilder


@dataclass
class GeneratedKernel:
    """Container for a single generated test-case pair.

    Both *triton_source* and *torch_ref_source* are complete, stand-alone
    Python modules that can be ``exec``'d independently.
    """

    seed: int
    triton_source: str          # Full Python source for the Triton kernel
    torch_ref_source: str       # Full Python source for the PyTorch reference
    triton_ast: ast.Module      # Parsed AST of the Triton source
    torch_ref_ast: ast.Module   # Parsed AST of the reference source
    metadata: dict = field(default_factory=dict)


class Generator:
    """Frontend that synthesises Triton + PyTorch source pairs.

    Each call to :meth:`generate` is fully deterministic given the same
    *seed*, so that any failure can be reproduced trivially.
    """

    def __init__(self, *, extra_config: Optional[dict] = None) -> None:
        self._extra_config = extra_config or {}
        self._max_body_ops: int | None = self._extra_config.get("max_body_ops")
        self._max_inputs: int | None = self._extra_config.get("max_inputs")

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, seed: int) -> GeneratedKernel:
        """Deterministically generate a ``(triton_kernel, torch_ref)`` pair.

        Parameters
        ----------
        seed:
            Integer seed that fully determines the generated code.

        Returns
        -------
        GeneratedKernel
        """
        rng = random.Random(seed)
        builder = KernelBuilder(seed, rng, extra_config=self._extra_config)
        triton_src, torch_src, metadata = builder.build()

        # Sanity-check: ensure both sources are valid Python.
        triton_ast_node = ast.parse(triton_src)
        torch_ast_node = ast.parse(torch_src)

        return GeneratedKernel(
            seed=seed,
            triton_source=triton_src,
            torch_ref_source=torch_src,
            triton_ast=triton_ast_node,
            torch_ref_ast=torch_ast_node,
            metadata=metadata,
        )
