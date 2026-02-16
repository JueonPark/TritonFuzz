"""Target Abstraction Layer – programmatic hardware target selection (§5.1).

Supports three resolution modes:

1. **Auto-detect** (``target_spec=None``):
   Queries ``triton.runtime.driver.active.get_current_target()``.
2. **Backend-only** (``target_spec="cuda"`` / ``"hip"`` / ``"cpu"``):
   Uses auto-detection but validates the backend matches.
3. **Explicit arch** (``target_spec="cuda:90"`` / ``"hip:gfx942"``):
   Constructs a ``GPUTarget`` for cross-compilation fuzzing — allows
   compiling for hardware you don't physically possess.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Data model                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #


@dataclass
class Target:
    """Lightweight, serialisable description of a compilation target."""

    backend: str    # "cuda" | "hip" | "cpu"
    arch: str       # e.g. "90" (sm_90), "gfx942", "x86_64"
    warp_size: int  # 32 for NVIDIA, 64 for AMD, 1 for CPU
    raw: Any = None  # Original GPUTarget object from Triton (or ``None``)

    def __str__(self) -> str:
        return f"{self.backend}:{self.arch} (warp_size={self.warp_size})"


# ── Default warp sizes per backend ────────────────────────────────────────

_WARP_SIZES: dict[str, int] = {
    "cuda": 32,
    "hip": 64,
    "cpu": 1,
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  TargetContext — resolution, caching, and option generation                #
# ═══════════════════════════════════════════════════════════════════════════ #


class TargetContext:
    """Resolves, caches, and exposes the active hardware target.

    Parameters
    ----------
    target_spec:
        One of:
        * ``None`` / ``"auto"`` — auto-detect the current device.
        * ``"cuda"`` / ``"hip"`` / ``"cpu"`` — auto-detect but assert backend.
        * ``"cuda:90"`` — build ``GPUTarget(backend="cuda", arch=90, warp_size=32)``.
        * ``"hip:gfx942"`` — build ``GPUTarget(backend="hip", arch="gfx942", warp_size=64)``.
    """

    def __init__(self, target_spec: Optional[str] = None) -> None:
        self._target_spec = target_spec
        self._target: Optional[Target] = None

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def target(self) -> Target:
        """Lazily resolve and return the current :class:`Target`."""
        if self._target is None:
            self._target = self._resolve_target()
            logger.info("Resolved target: %s", self._target)
        return self._target

    def get_backend_options(self, rng: random.Random | None = None) -> dict:
        """Return backend-specific compilation options.

        When *rng* is provided the values are randomised (deterministically);
        otherwise sensible defaults are returned.
        """
        t = self.target
        if t.backend == "cuda":
            return self._generate_cuda_options(rng)
        elif t.backend == "hip":
            return self._generate_hip_options(rng)
        elif t.backend == "cpu":
            return self._generate_cpu_options(rng)
        return {}

    def get_triton_target(self) -> Any:
        """Return the raw Triton ``GPUTarget`` (or equivalent) for ``triton.compile()``.

        Falls back to constructing one from our :class:`Target` fields if the
        raw handle is unavailable (cross-compilation scenario).
        """
        t = self.target
        if t.raw is not None:
            return t.raw
        return self._construct_triton_target(t)

    # ── Resolution logic (§5.1) ──────────────────────────────────────────

    def _resolve_target(self) -> Target:
        spec = self._target_spec

        # Auto-detect
        if spec is None or spec == "auto":
            return self._auto_detect()

        # "cuda", "hip", "cpu" — backend only (no arch)
        if ":" not in spec:
            backend = spec.lower()
            detected = self._auto_detect()
            if detected.backend == backend:
                return detected
            # Backend mismatch → cross-compilation mode
            logger.warning(
                "Requested backend '%s' but detected '%s' → cross-compilation mode",
                backend,
                detected.backend,
            )
            return self._make_synthetic(backend, arch=None)

        # "cuda:90", "hip:gfx942" — explicit arch
        backend, arch = spec.split(":", maxsplit=1)
        backend = backend.lower()
        return self._make_synthetic(backend, arch)

    def _auto_detect(self) -> Target:
        """Query Triton for the currently-active device."""
        try:
            import triton

            raw = triton.runtime.driver.active.get_current_target()
            return Target(
                backend=raw.backend,
                arch=str(raw.arch),
                warp_size=getattr(raw, "warp_size", _WARP_SIZES.get(raw.backend, 32)),
                raw=raw,
            )
        except Exception as exc:
            logger.warning("Auto-detection failed (%s), falling back to cpu", exc)
            return Target(backend="cpu", arch="x86_64", warp_size=1)

    def _make_synthetic(self, backend: str, arch: Optional[str]) -> Target:
        """Build a synthetic Target for cross-compilation."""
        warp_size = _WARP_SIZES.get(backend, 32)

        if backend == "cuda":
            arch = arch or "90"
        elif backend == "hip":
            arch = arch or "gfx942"
        elif backend == "cpu":
            arch = arch or "x86_64"
            warp_size = 1
        else:
            arch = arch or "unknown"

        return Target(backend=backend, arch=arch, warp_size=warp_size)

    @staticmethod
    def _construct_triton_target(t: Target) -> Any:
        """Attempt to build a real Triton ``GPUTarget`` from our fields.

        Used when the raw handle is unavailable (cross-compilation from a
        machine without the target hardware).
        """
        try:
            if t.backend == "cuda":
                from triton.backends.nvidia.driver import GPUTarget

                return GPUTarget(backend="cuda", arch=int(t.arch), warp_size=t.warp_size)
            elif t.backend == "hip":
                from triton.backends.amd.driver import GPUTarget

                return GPUTarget(backend="hip", arch=t.arch, warp_size=t.warp_size)
        except Exception as exc:
            logger.debug("Could not construct GPUTarget: %s", exc)
        return None

    # ── Backend option generators (§5.2 – §5.4) ─────────────────────────

    @staticmethod
    def _generate_cuda_options(rng: random.Random | None) -> dict:
        """CUDA-specific compilation options (§5.2).

        Fuzzed parameters:
        - ``num_warps``: changes thread layout; bank conflicts at high counts.
        - ``num_stages``: software pipelining depth (cp.async on Ampere+).
        - ``enable_fp_fusion``: toggles FMA fusion of mul + add.
        """
        if rng is None:
            return {"num_warps": 4, "num_stages": 2, "enable_fp_fusion": True}
        return {
            "num_warps": rng.choice([1, 2, 4, 8]),
            "num_stages": rng.choice([1, 2, 3, 4, 5]),
            "enable_fp_fusion": rng.choice([True, False]),
        }

    @staticmethod
    def _generate_hip_options(rng: random.Random | None) -> dict:
        """HIP/ROCm-specific compilation options (§5.3).

        Fuzzed parameters:
        - ``num_warps``: wavefronts (64 threads each on CDNA).
        - ``waves_per_eu``: occupancy control for MI200/MI300.
        - ``matrix_core_version``: toggles MFMA vs. fallback VALU.
        """
        if rng is None:
            return {
                "num_warps": 4,
                "waves_per_eu": 1,
                "matrix_core_version": 0,
            }
        return {
            "num_warps": rng.choice([1, 2, 4, 8]),
            "waves_per_eu": rng.choice([1, 2, 4]),
            "matrix_core_version": rng.choice([0, 1, 2]),
        }

    @staticmethod
    def _generate_cpu_options(rng: random.Random | None) -> dict:
        """CPU-specific compilation options (§5.4).

        The triton-cpu backend has very few knobs today; the main testing
        strategy is to verify vectorisation (checked externally via
        ``LLVM_IR_ENABLE_DUMP=1``).
        """
        return {}
