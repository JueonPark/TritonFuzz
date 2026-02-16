"""Generator package – synthesises valid Triton kernels and PyTorch references.

Public re-exports::

    from tritonfuzz.generator import Generator, GeneratedKernel
"""

from tritonfuzz.generator.core import GeneratedKernel, Generator

__all__ = ["Generator", "GeneratedKernel"]
