"""Reducer – minimises a failing test-case via AST-based delta debugging (§7).

When the Oracle reports a FAIL, CRASH, or NAN_MISMATCH, the Reducer shrinks
the generated kernel while preserving the failure, producing a minimal
reproducer that is actionable for developers.

§7.1 — AST-Based Delta Debugging
  - **Operation Pruning**: iteratively removes body operations from the
    kernel DAG.  If the bug persists the operation was irrelevant.
  - **Dimension Reduction**: shrinks ``BLOCK_SIZE`` and ``n_elements``.
  - **Type Simplification**: demotes ``bfloat16`` / ``float16`` → ``float32``
    to determine whether the bug is precision-specific.

§7.2 — Artifact Collection
  - Sets diagnostic environment variables (``MLIR_ENABLE_DUMP``, etc.)
    during a reproduction run.
  - Captures IR dumps and the minimal Python reproduction script.
  - Packages everything into a zip file for developer analysis.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tritonfuzz.config import FuzzConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Result types                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #


@dataclass
class ReductionResult:
    """Outcome of a reduction attempt."""

    seed: int
    original_source: str
    reduced_source: str
    reduced_ref_source: str
    original_line_count: int
    reduced_line_count: int
    steps_taken: list[str] = field(default_factory=list)
    artifact_path: Optional[Path] = None
    precision_specific: bool = False  # True if bug disappears with float32


# ═══════════════════════════════════════════════════════════════════════════ #
#  Source-line analysis helpers                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

# Regex patterns for classifying lines in the generated sources.
_VAR_DEF_RE = re.compile(r"^(\s*)(v_\d+)\s*=\s*(.+)$")
_VAR_REF_RE = re.compile(r"\bv_\d+\b")
_LOOP_HEADER_RE = re.compile(r"^(\s*)for\s+_loop_i\s+in\s+range\(\d+\):")
_LOAD_CALL_RE = re.compile(r"\btl\.load\b")
_STORE_CALL_RE = re.compile(r"\btl\.store\b")
_PREAMBLE_RE = re.compile(r"^\s*(pid|block_start|offsets|mask)\s*=")
_INPUT_ASSIGN_RE = re.compile(r"^\s*v_\d+\s*=\s*x\d+\s*$")

# Type simplification maps (§7.1).
_DTYPE_REPLACEMENTS: list[tuple[str, str]] = [
    ("tl.bfloat16", "tl.float32"),
    ("tl.float16", "tl.float32"),
    ("torch.bfloat16", "torch.float32"),
    ("torch.float16", "torch.float32"),
]

# Dimension reduction ladders (largest → smallest).
_BLOCK_SIZE_LADDER: list[int] = [1024, 512, 256, 128, 64]
_N_ELEMENTS_LADDER: list[int] = [8192, 4096, 2048, 1024, 512, 256]

# §7.2 — Environment variables for artifact collection.
_ARTIFACT_ENV_VARS: dict[str, str] = {
    "MLIR_ENABLE_DUMP": "1",
    "TRITON_ALWAYS_COMPILE": "1",
    "LLVM_IR_ENABLE_DUMP": "1",
}

# Backend-specific dump variables (§7.2).
_BACKEND_DUMP_VARS: dict[str, dict[str, str]] = {
    "cuda": {"NVPTX_ENABLE_DUMP": "1"},
    "hip": {"AMD_GCN_ENABLE_DUMP": "1"},
    "cpu": {},
}


# ── Internal dataclass for body-op tracking ──────────────────────────────


@dataclass
class _BodyOp:
    """A single removable body operation in the kernel source."""

    line_indices: list[int]   # 0-based line numbers this op spans
    defines: str              # Variable name defined (e.g. "v_3")
    uses: list[str]           # Variables referenced on the RHS
    is_loop: bool = False     # True if this is a for-loop block


# ── Classification ───────────────────────────────────────────────────────


def _classify_body_ops(lines: list[str]) -> list[_BodyOp]:
    """Identify removable body operations in kernel source lines.

    Returns a list of :class:`_BodyOp` descriptors.  Imports, the function
    signature, preamble (``pid``, ``offsets``, ``mask``), loads, store,
    return, and simple input assignments (``v_0 = x0``) are excluded.
    """
    ops: list[_BodyOp] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- Skip non-body lines -----------------------------------------
        if (
            not stripped
            or stripped.startswith(("import ", "from ", "@", "def ", "#", ")"))
        ):
            i += 1
            continue
        if _PREAMBLE_RE.match(line):
            i += 1
            continue
        if _STORE_CALL_RE.search(line):
            i += 1
            continue
        if stripped.startswith("return "):
            i += 1
            continue
        if _LOAD_CALL_RE.search(line):
            i += 1
            continue
        if _INPUT_ASSIGN_RE.match(line):
            i += 1
            continue

        # --- For-loop block ----------------------------------------------
        loop_match = _LOOP_HEADER_RE.match(line)
        if loop_match:
            indent_level = len(loop_match.group(1))
            loop_indices = [i]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt_indent > indent_level:
                    loop_indices.append(i)
                    i += 1
                else:
                    break

            all_defs: set[str] = set()
            all_refs: set[str] = set()
            for li in loop_indices:
                m = _VAR_DEF_RE.match(lines[li])
                if m:
                    all_defs.add(m.group(2))
                all_refs.update(_VAR_REF_RE.findall(lines[li]))

            if all_defs:
                external_uses = [u for u in all_refs if u not in all_defs]
                ops.append(_BodyOp(
                    line_indices=loop_indices,
                    defines=sorted(all_defs)[0],
                    uses=external_uses,
                    is_loop=True,
                ))
            continue  # i already past the loop block

        # --- Regular body op: v_N = <expr> -------------------------------
        m = _VAR_DEF_RE.match(line)
        if m:
            var_name = m.group(2)
            rhs = m.group(3)
            uses = _VAR_REF_RE.findall(rhs)
            ops.append(_BodyOp(
                line_indices=[i],
                defines=var_name,
                uses=uses,
            ))

        i += 1

    return ops


# ── Source rewriting ─────────────────────────────────────────────────────


def _remove_op_from_source(
    lines: list[str],
    op: _BodyOp,
    replacement_var: str,
) -> list[str]:
    """Remove an op's lines and substitute the defined variable if needed.

    If the variable is still defined in the remaining code (e.g. by an
    earlier initializer), no substitution is performed — the earlier
    definition supplies the value.  Otherwise every reference to the removed
    variable is replaced with *replacement_var*.
    """
    removed = set(op.line_indices)
    remaining = [lines[i] for i in range(len(lines)) if i not in removed]

    # Check if the variable is still defined somewhere in the remaining lines
    pat = re.compile(r"^\s*" + re.escape(op.defines) + r"\s*=")
    still_defined = any(pat.match(ln) for ln in remaining)

    if still_defined:
        return remaining

    # Variable would be undefined → substitute all references
    sub_re = re.compile(r"\b" + re.escape(op.defines) + r"\b")
    return [sub_re.sub(replacement_var, ln) for ln in remaining]


def _is_valid_python(source: str) -> bool:
    """Return *True* if *source* parses without syntax errors."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


# ═══════════════════════════════════════════════════════════════════════════ #
#  Reducer                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


class Reducer:
    """Minimises a failing kernel to the smallest reproducing example (§7).

    The Reducer requires access to the Driver, Runtime, and Oracle so that
    it can re-compile, re-execute, and re-verify each candidate reduction.
    If any of these are ``None``, :meth:`reduce` logs a warning and returns
    ``None``.
    """

    def __init__(
        self,
        config: FuzzConfig,
        *,
        driver: Any = None,
        runtime: Any = None,
        oracle: Any = None,
    ) -> None:
        self._config = config
        self._driver = driver
        self._runtime = runtime
        self._oracle = oracle

    # ── Public API ────────────────────────────────────────────────────────

    def reduce(
        self,
        kernel,           # GeneratedKernel
        compiled,         # CompilationResult
        exec_result,      # ExecutionResult
        verdict=None,     # VerificationResult — original failing verdict
    ) -> Optional[ReductionResult]:
        """Attempt to shrink a failing test-case.

        Parameters
        ----------
        kernel : GeneratedKernel
            The original generated kernel that triggered the failure.
        compiled : CompilationResult
            The compilation result (carries compile options to reuse).
        exec_result : ExecutionResult
            The execution result (for context; not directly mutated).
        verdict : VerificationResult, optional
            The Oracle's verdict.  If provided, the reducer preserves this
            specific verdict kind during delta-debugging.

        Returns
        -------
        ReductionResult or None
            The reduction outcome, or ``None`` if reduction was skipped.
        """
        if self._driver is None or self._runtime is None or self._oracle is None:
            logger.warning(
                "Reducer for seed %d: driver/runtime/oracle not provided – "
                "skipping reduction",
                kernel.seed,
            )
            return None

        from tritonfuzz.oracle import Verdict

        target_verdict = verdict.verdict if verdict is not None else Verdict.FAIL

        logger.info(
            "Reducing seed %d (target: %s)", kernel.seed, target_verdict.value,
        )

        triton_src: str = kernel.triton_source
        torch_src: str = kernel.torch_ref_source
        metadata: dict = dict(kernel.metadata)
        compile_opts: dict = dict(compiled.compile_options)
        steps: list[str] = []
        original_lines = len(triton_src.splitlines())

        # ── §7.1 Phase 1: Operation Pruning ──────────────────────────────
        triton_src, torch_src, p_steps = self._prune_operations(
            triton_src, torch_src, metadata, compile_opts, target_verdict,
        )
        steps.extend(p_steps)

        # ── §7.1 Phase 2: Dimension Reduction ────────────────────────────
        triton_src, torch_src, metadata, d_steps = self._reduce_dimensions(
            triton_src, torch_src, metadata, compile_opts, target_verdict,
        )
        steps.extend(d_steps)

        # ── §7.1 Phase 3: Type Simplification ────────────────────────────
        triton_src, torch_src, metadata, t_steps, prec = self._simplify_types(
            triton_src, torch_src, metadata, compile_opts, target_verdict,
        )
        steps.extend(t_steps)

        # ── §7.2 Artifact Collection ─────────────────────────────────────
        artifact_path = self._collect_artifacts(
            kernel.seed, triton_src, torch_src, metadata, compile_opts,
        )
        if artifact_path is not None:
            steps.append(f"Artifacts → {artifact_path}")

        reduced_lines = len(triton_src.splitlines())
        logger.info(
            "Reduction for seed %d: %d → %d lines (%d steps)",
            kernel.seed, original_lines, reduced_lines, len(steps),
        )

        return ReductionResult(
            seed=kernel.seed,
            original_source=kernel.triton_source,
            reduced_source=triton_src,
            reduced_ref_source=torch_src,
            original_line_count=original_lines,
            reduced_line_count=reduced_lines,
            steps_taken=steps,
            artifact_path=artifact_path,
            precision_specific=prec,
        )

    # ── Re-test predicate ────────────────────────────────────────────────

    def _reproduces(
        self,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
        target_verdict: Any,
    ) -> bool:
        """Return ``True`` if the modified sources still trigger the bug.

        Performs the full pipeline: compile → execute → verify, and checks
        whether the resulting verdict matches *target_verdict*.
        """
        from tritonfuzz.generator import GeneratedKernel
        from tritonfuzz.oracle import Verdict

        if not _is_valid_python(triton_src) or not _is_valid_python(torch_src):
            return False

        try:
            mod_kernel = GeneratedKernel(
                seed=metadata.get("seed", 0),
                triton_source=triton_src,
                torch_ref_source=torch_src,
                triton_ast=ast.parse(triton_src),
                torch_ref_ast=ast.parse(torch_src),
                metadata=metadata,
            )

            comp = self._driver.compile(mod_kernel)

            # Special case: target is COMPILE_ERROR — we check compilation failure
            if target_verdict == Verdict.COMPILE_ERROR:
                return not comp.success
            if not comp.success:
                return False  # Unexpected compilation failure

            er = self._runtime.execute(mod_kernel, comp)
            vr = self._oracle.verify(er)
            return vr.verdict == target_verdict

        except Exception as exc:  # noqa: BLE001
            logger.debug("Re-test exception: %s", exc)
            return False

    # ── Phase 1: Operation Pruning (§7.1) ─────────────────────────────────

    def _prune_operations(
        self,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
        target_verdict: Any,
    ) -> tuple[str, str, list[str]]:
        """Iteratively remove body ops while preserving the bug.

        Uses a 1-minimal delta-debugging strategy: try removing each op
        one at a time (in reverse order), keep the removal if the bug
        persists, then restart until no more removals are possible.

        Respects ``FuzzConfig.max_reduction_steps`` to bound the total
        number of re-test invocations across all pruning passes.
        """
        steps: list[str] = []
        changed = True
        budget = self._config.max_reduction_steps

        while changed and budget > 0:
            changed = False
            triton_lines = triton_src.splitlines()
            torch_lines = torch_src.splitlines()

            body_ops = _classify_body_ops(triton_lines)
            if not body_ops:
                break

            # Try removing each op in reverse (later ops are more likely
            # to be safely removable because fewer things depend on them).
            for op in reversed(body_ops):
                if not op.uses:
                    continue  # Cannot substitute without an input variable

                replacement = op.uses[0]

                # Remove from Triton source
                new_tl = _remove_op_from_source(triton_lines, op, replacement)
                new_triton = "\n".join(new_tl)

                # Find and remove the matching op in the torch ref
                torch_ops = _classify_body_ops(torch_lines)
                match = next(
                    (t for t in torch_ops if t.defines == op.defines), None,
                )
                if match is not None:
                    new_rl = _remove_op_from_source(
                        torch_lines, match, replacement,
                    )
                    new_torch = "\n".join(new_rl)
                else:
                    new_torch = torch_src

                # Quick syntax gate (avoids expensive re-compile)
                if not _is_valid_python(new_triton):
                    continue
                if match is not None and not _is_valid_python(new_torch):
                    continue

                budget -= 1
                if budget <= 0:
                    steps.append("Pruning budget exhausted")
                    break

                if self._reproduces(
                    new_triton, new_torch, metadata,
                    compile_options, target_verdict,
                ):
                    triton_src = new_triton
                    torch_src = new_torch
                    label = "loop" if op.is_loop else "op"
                    msg = f"Pruned {label}: {op.defines}"
                    steps.append(msg)
                    logger.debug(
                        "seed %s: %s", metadata.get("seed", "?"), msg,
                    )
                    changed = True
                    break  # Restart with updated source

        return triton_src, torch_src, steps

    # ── Phase 2: Dimension Reduction (§7.1) ───────────────────────────────

    def _reduce_dimensions(
        self,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
        target_verdict: Any,
    ) -> tuple[str, str, dict, list[str]]:
        """Shrink ``BLOCK_SIZE`` and ``n_elements`` while preserving the bug."""
        steps: list[str] = []
        cur_bs = metadata.get("block_size", 256)
        cur_ne = metadata.get("n_elements", 1024)

        # Try successively smaller BLOCK_SIZE values
        for bs in _BLOCK_SIZE_LADDER:
            if bs >= cur_bs:
                continue
            cand = dict(metadata, block_size=bs)
            if self._reproduces(
                triton_src, torch_src, cand, compile_options, target_verdict,
            ):
                cur_bs = bs
                metadata = cand
                msg = f"BLOCK_SIZE → {bs}"
                steps.append(msg)
                logger.debug("seed %s: %s", metadata.get("seed", "?"), msg)

        # Try successively smaller n_elements values
        for ne in _N_ELEMENTS_LADDER:
            if ne >= cur_ne:
                continue
            cand = dict(metadata, n_elements=ne)
            if self._reproduces(
                triton_src, torch_src, cand, compile_options, target_verdict,
            ):
                cur_ne = ne
                metadata = cand
                msg = f"n_elements → {ne}"
                steps.append(msg)
                logger.debug("seed %s: %s", metadata.get("seed", "?"), msg)

        return triton_src, torch_src, metadata, steps

    # ── Phase 3: Type Simplification (§7.1) ───────────────────────────────

    def _simplify_types(
        self,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
        target_verdict: Any,
    ) -> tuple[str, str, dict, list[str], bool]:
        """Demote ``bfloat16`` / ``float16`` → ``float32``.

        Returns
        -------
        tuple
            ``(triton_src, torch_src, metadata, steps, precision_specific)``
            where *precision_specific* is ``True`` when the bug disappears
            after simplification (the mismatch is due to narrow-type
            precision, not a compiler defect).
        """
        steps: list[str] = []

        # Only attempt if narrow types are present in the source
        has_narrow = any(old in triton_src for old, _ in _DTYPE_REPLACEMENTS[:2])
        if not has_narrow:
            return triton_src, torch_src, metadata, steps, False

        # Build candidate with all narrow types promoted
        new_triton = triton_src
        new_torch = torch_src
        for old, new in _DTYPE_REPLACEMENTS:
            new_triton = new_triton.replace(old, new)
            new_torch = new_torch.replace(old, new)

        cand_meta = dict(metadata)
        cand_meta["input_dtypes"] = [
            "torch.float32" if d in ("torch.float16", "torch.bfloat16") else d
            for d in cand_meta.get("input_dtypes", [])
        ]
        if cand_meta.get("output_dtype") in ("torch.float16", "torch.bfloat16"):
            cand_meta["output_dtype"] = "torch.float32"

        if self._reproduces(
            new_triton, new_torch, cand_meta, compile_options, target_verdict,
        ):
            steps.append("Types simplified: bf16/fp16 → fp32 (NOT precision-specific)")
            logger.debug(
                "seed %s: types simplified, bug persists",
                metadata.get("seed", "?"),
            )
            return new_triton, new_torch, cand_meta, steps, False
        else:
            steps.append("Types: bug vanishes with fp32 (precision-specific)")
            logger.debug(
                "seed %s: bug disappears with fp32 → precision-specific",
                metadata.get("seed", "?"),
            )
            return triton_src, torch_src, metadata, steps, True

    # ── §7.2: Artifact Collection ─────────────────────────────────────────

    def _collect_artifacts(
        self,
        seed: int,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
    ) -> Optional[Path]:
        """Write reproducer, IR dumps, and metadata; package into a zip.

        Returns the path to the zip file, or the output directory on
        failure, or ``None`` if the directory could not be created.
        """
        out_dir = self._config.output_dir / f"seed_{seed}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot create artifact dir: %s", exc)
            return None

        # ── Standalone reproduction script ────────────────────────────────
        repro = self._build_repro_script(
            seed, triton_src, torch_src, metadata, compile_options,
        )
        (out_dir / "reproduce.py").write_text(repro)

        # ── Individual source files ───────────────────────────────────────
        (out_dir / "triton_kernel.py").write_text(triton_src)
        (out_dir / "torch_ref.py").write_text(torch_src)

        # ── Metadata JSON ─────────────────────────────────────────────────
        serialisable = {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        serialisable["compile_options"] = compile_options
        (out_dir / "metadata.json").write_text(
            json.dumps(serialisable, indent=2) + "\n",
        )

        # ── IR dump collection ────────────────────────────────────────────
        ir_dir = out_dir / "ir_dumps"
        ir_dir.mkdir(exist_ok=True)
        self._collect_ir_dumps(seed, triton_src, metadata, compile_options, ir_dir)

        # ── Package into a zip ────────────────────────────────────────────
        zip_path = self._config.output_dir / f"seed_{seed}_artifacts.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in out_dir.rglob("*"):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(self._config.output_dir))
            logger.info("Artifacts for seed %d → %s", seed, zip_path)
            return zip_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Zip creation failed for seed %d: %s", seed, exc)
            return out_dir  # fall back to the directory itself

    def _collect_ir_dumps(
        self,
        seed: int,
        triton_src: str,
        metadata: dict,
        compile_options: dict,
        dump_dir: Path,
    ) -> None:
        """Re-compile with diagnostic env vars to capture IR dumps (§7.2).

        Sets:
          ``MLIR_ENABLE_DUMP=1``
          ``TRITON_ALWAYS_COMPILE=1``
          ``LLVM_IR_ENABLE_DUMP=1``
          ``NVPTX_ENABLE_DUMP=1`` (CUDA) / ``AMD_GCN_ENABLE_DUMP=1`` (HIP)
        """
        # Determine backend from the driver's target context
        backend = "cuda"
        if (
            hasattr(self._driver, "_target_ctx")
            and self._driver._target_ctx.target is not None
        ):
            backend = self._driver._target_ctx.target.backend

        env_vars = dict(_ARTIFACT_ENV_VARS)
        env_vars.update(_BACKEND_DUMP_VARS.get(backend, {}))

        # Launch compilation in a subprocess to capture stdout/stderr
        # (Triton and LLVM dump to stdout/stderr when the env vars are set).
        wrapper = (
            "import ast, sys, json\n"
            "from tritonfuzz.generator import GeneratedKernel\n"
            "from tritonfuzz.driver import Driver\n"
            "from tritonfuzz.config import FuzzConfig\n"
            f"meta = {metadata!r}\n"
            f"src = {triton_src!r}\n"
            "mk = GeneratedKernel(\n"
            f"    seed={seed},\n"
            "    triton_source=src,\n"
            "    torch_ref_source='',\n"
            "    triton_ast=ast.parse(src),\n"
            "    torch_ref_ast=ast.parse('pass'),\n"
            "    metadata=meta,\n"
            ")\n"
            "cfg = FuzzConfig()\n"
            "drv = Driver(cfg)\n"
            "comp = drv.compile(mk)\n"
            "print('SUCCESS' if comp.success else 'FAIL', file=sys.stderr)\n"
        )

        sub_env = {**os.environ, **env_vars}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=60,
                env=sub_env,
            )
            stdout_log = proc.stdout
            stderr_log = proc.stderr
            status = f"returncode={proc.returncode}"
        except subprocess.TimeoutExpired:
            stdout_log = ""
            stderr_log = ""
            status = "subprocess timed out"
        except Exception as exc:  # noqa: BLE001
            stdout_log = ""
            stderr_log = ""
            status = f"exception: {exc}"

        (dump_dir / "compile_status.txt").write_text(
            f"Compilation {status}\n"
            f"Environment: {env_vars}\n"
            f"Options: {compile_options}\n",
        )
        if stdout_log:
            (dump_dir / "stdout.log").write_text(stdout_log)
        if stderr_log:
            (dump_dir / "stderr.log").write_text(stderr_log)

    # ── Reproducer script generation ──────────────────────────────────────

    @staticmethod
    def _build_repro_script(
        seed: int,
        triton_src: str,
        torch_src: str,
        metadata: dict,
        compile_options: dict,
    ) -> str:
        """Generate a standalone Python script that reproduces the bug.

        The script embeds both the Triton kernel and the PyTorch reference,
        allocates inputs according to metadata, and runs the comparison.
        """
        bs = metadata.get("block_size", 256)
        ne = metadata.get("n_elements", 1024)
        n_in = metadata.get("num_inputs", 1)
        kfn = metadata.get("kernel_fn_name", f"triton_kernel_seed_{seed}")
        rfn = metadata.get("ref_fn_name", f"torch_ref_seed_{seed}")
        dtypes = metadata.get("input_dtypes", ["torch.float32"] * n_in)

        # Build input allocation lines
        allocs: list[str] = []
        names: list[str] = []
        for i in range(n_in):
            dt = dtypes[i] if i < len(dtypes) else "torch.float32"
            nm = f"x{i}"
            names.append(nm)
            allocs.append(
                f'{nm} = torch.randn({ne}, device="cuda", dtype={dt})'
            )

        alloc_block = "\n    ".join(allocs)
        ptr_args = ", ".join(names)
        ref_args = ", ".join(names)
        out_dt = dtypes[0] if dtypes else "torch.float32"
        grid_expr = f"({ne} + {bs} - 1) // {bs}"

        # Build the script as plain strings (no nested f-strings)
        parts: list[str] = [
            "#!/usr/bin/env python3\n",
            f'"""Minimal reproducer for TritonFuzz seed {seed}.\n',
            "\n",
            "To capture full IR dumps, run with:\n",
            "  MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \\\n",
            "  LLVM_IR_ENABLE_DUMP=1 \\\n",
            "  NVPTX_ENABLE_DUMP=1 python reproduce.py\n",
            '"""\n',
            "import torch\n",
            "\n",
            "# ── Triton kernel "
            "────────────────────────────────────────────────\n",
            triton_src,
            "\n",
            "# ── PyTorch reference "
            "────────────────────────────────────────────\n",
            torch_src,
            "\n\n",
            "def main():\n",
            f"    {alloc_block}\n",
            f"    out = torch.empty({ne}, "
            f'device="cuda", dtype={out_dt})\n',
            "\n",
            "    # Golden output\n",
            f"    golden = {rfn}({ref_args})\n",
            "\n",
            "    # Triton output\n",
            f"    grid = ({grid_expr},)\n",
            f"    {kfn}[grid]({ptr_args}, out, {ne}, BLOCK_SIZE={bs})\n",
            "\n",
            "    # Compare\n",
            "    if torch.allclose(golden, out, atol=1e-2, rtol=1e-2):\n",
            '        print("PASS")\n',
            "    else:\n",
            "        diff = (golden.float() - out.float()).abs()\n",
            '        print(f"FAIL — max_abs_diff:'
            ' {diff.max().item():.6e}")\n',
            '        print(f"  golden[:8]: {golden[:8]}")\n',
            '        print(f"  triton[:8]: {out[:8]}")\n',
            "\n\n",
            'if __name__ == "__main__":\n',
            "    main()\n",
        ]
        return "".join(parts)
