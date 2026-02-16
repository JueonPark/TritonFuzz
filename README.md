# TritonFuzz

A differential-testing fuzzer for the [Triton](https://github.com/triton-lang/triton) compiler.
TritonFuzz automatically generates random Triton GPU kernels, compiles them across backends (CUDA, HIP, CPU), executes them alongside PyTorch reference implementations, and reports miscompilations, crashes, and NaN-propagation bugs.

## Overview

TritonFuzz implements a five-stage pipeline that runs on every seed:

```
Generator → Driver → Runtime → Oracle → Reducer
```

| Stage | Module | What it does |
|---|---|---|
| **Generator** | `generator/` | Synthesises a random `@triton.jit` kernel and a matching PyTorch reference function from a single integer seed. Each seed produces the same kernel deterministically. |
| **Driver** | `driver.py` | Compiles the Triton kernel via `triton.compile()` with (optionally randomised) backend options. Supports config sweeping to detect divergence bugs across tuning configs. |
| **Runtime** | `runtime.py` | Allocates input/output tensors on GPU, executes both the PyTorch reference (golden output) and the compiled Triton kernel (test output), with per-kernel timeout protection. |
| **Oracle** | `oracle.py` | Compares golden and test outputs using dynamic tolerances adjusted for dtype and operation complexity. Detects numerical mismatches, NaN/Inf propagation bugs, and (for atomic ops) global-sum/histogram discrepancies. |
| **Reducer** | `reducer.py` | When a bug is found, minimises the failing kernel via AST-based delta debugging (operation pruning, dimension reduction, type simplification) and packages a standalone reproducer with IR dumps. |

## Installation

Requires **Python 3.10+**, a working [Triton](https://github.com/triton-lang/triton) install, and [PyTorch](https://pytorch.org/).

```bash
# From the repo root
pip install -e .

# With development tools
pip install -e ".[dev]"
```

## Quick start

```bash
# Fuzz seeds 0–99 on the auto-detected GPU
tritonfuzz --seed-start 0 --seed-end 100

# Target a specific CUDA architecture (cross-compilation)
tritonfuzz --seed-start 0 --seed-end 500 --target cuda:90

# Sweep 5 compile-option variants per seed to detect divergence bugs
tritonfuzz --seed-end 200 --sweep-count 5

# Increase verbosity
tritonfuzz --seed-end 50 -vv
```

### Programmatic usage

```python
from tritonfuzz.config import FuzzConfig
from tritonfuzz.fuzzer import Fuzzer

config = FuzzConfig(
    seed_start=0,
    seed_end=500,
    target_spec="cuda",
    sweep_count=0,
    atol=1e-2,
    rtol=1e-2,
)

fuzzer = Fuzzer(config)
stats = fuzzer.run()
print(stats.summary())
# Total: 500 | Pass: 487 | Fail: 8 | Crash: 2 | Timeout: 0 | CompileErr: 3 | NaN: 0
```

### Generate a single kernel (without running the full pipeline)

```python
from tritonfuzz.generator import Generator

gen = Generator()
kernel = gen.generate(seed=42)

print(kernel.triton_source)   # Complete @triton.jit kernel
print(kernel.torch_ref_source) # Matching PyTorch function
print(kernel.metadata)         # Dict with dtypes, block_size, ops_used, etc.
```

## CLI reference

```
usage: tritonfuzz [-h] [--seed-start N] [--seed-end N] [--target SPEC]
                  [--sweep-count N] [--no-randomize-options]
                  [--atol F] [--rtol F] [--output-dir DIR] [-v]

Options:
  --seed-start N          First seed (default: 0)
  --seed-end N            One-past-the-last seed (default: 100)
  --target SPEC           Target: auto, cuda, hip, cpu, cuda:90, hip:gfx942
  --sweep-count N         Compile each seed N times with different options
                          and cross-check for divergence (default: 0 = off)
  --no-randomize-options  Use default compile options instead of randomising
  --atol F                Absolute tolerance (default: 1e-2)
  --rtol F                Relative tolerance (default: 1e-2)
  --output-dir DIR        Directory for results & artifacts (default: tritonfuzz_results)
  -v, --verbose           -v for INFO, -vv for DEBUG
```

## Configuration

All knobs are fields on `FuzzConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `seed_start` | `int` | `0` | First seed in the range |
| `seed_end` | `int` | `1000` | One-past-last seed |
| `seed_list` | `list[int] \| None` | `None` | Explicit seed list (overrides range) |
| `target_spec` | `str \| None` | `None` | Target specifier (see CLI `--target`) |
| `randomize_compile_options` | `bool` | `True` | Randomise backend options per seed |
| `sweep_count` | `int` | `0` | Config-sweep variants per seed |
| `timeout_seconds` | `float` | `30.0` | Per-kernel execution timeout |
| `atol` | `float` | `1e-2` | Base absolute tolerance |
| `rtol` | `float` | `1e-2` | Base relative tolerance |
| `output_dir` | `Path` | `tritonfuzz_results` | Artifact output directory |
| `reduce_on_failure` | `bool` | `True` | Run the reducer on failures |
| `max_reduction_steps` | `int` | `50` | Budget for reduction re-tests |

## Architecture

```
src/tritonfuzz/
├── __init__.py
├── cli.py              # CLI entry point
├── config.py           # FuzzConfig dataclass
├── fuzzer.py           # Main loop (orchestrates the pipeline)
├── driver.py           # CompilerDriver – triton.compile() wrapper
├── runtime.py          # Runtime – tensor allocation & kernel launch
├── oracle.py           # Oracle – differential verification
├── reducer.py          # Reducer – AST-based delta debugging
├── target.py           # TargetContext – hardware target resolution
└── generator/
    ├── __init__.py
    ├── core.py          # Generator & GeneratedKernel
    ├── builder.py       # KernelBuilder – 5-phase synthesis engine
    ├── ops.py           # OpTemplate registry (unary, binary, logic, cast)
    ├── symbol_table.py  # SymbolTable & TensorVar
    └── types.py         # DType registry & promotion rules
```

### Generator

The **KernelBuilder** constructs kernels in five phases:

1. **Plan** — decide number of inputs, dtypes, body-op count, block size, whether to insert a for-loop, and whether to mix dtypes.
2. **Emit loads** — generate `tl.load()` calls with optional masking and `other=` values.
3. **Emit body** — build a randomised DAG of element-wise (unary/binary), logic (`tl.where`, `tl.minimum`, `tl.maximum`), and type-cast operations. Optionally wraps part of the body in a `for`-loop to stress LICM and software pipelining.
4. **Choose output** — select the last computed variable as the store target.
5. **Assemble** — emit the complete `@triton.jit` kernel and the matching `def torch_ref_seed_N(...)` function.

Supported operation categories:

| Category | Examples | Fuzzing motivation |
|---|---|---|
| Element-wise unary | `exp`, `sin`, `cos`, `log`, `abs`, `sqrt`, `neg` | Chain deeply to exhaust registers; mix dtypes |
| Element-wise binary | `+`, `-`, `*`, `/` | Implicit type promotion stress |
| Logic | `tl.where`, `tl.minimum`, `tl.maximum` | Predicate generation, select lowering |
| Type cast | `.to(tl.float16)`, `.to(tl.int32)`, … | Cast lowering across all dtype pairs |
| For-loop | `for _loop_i in range(K): acc = acc ⊕ x` | LICM, software pipelining |

### Driver

- Supports **CUDA** (`num_warps`, `num_stages`, `enable_fp_fusion`), **HIP** (`num_warps`, `waves_per_eu`, `matrix_core_version`), and **CPU** backends.
- Three target resolution modes: auto-detect, backend-only, explicit arch (cross-compilation).
- **Config sweeping**: compiles the same kernel N times with different options; the fuzzer cross-checks outputs for divergence bugs.
- Builds per-dtype `triton.compile()` signatures from kernel metadata.

### Oracle

- **Dynamic tolerances**: adjusted per-kernel based on output dtype (FP16/BF16 get looser bounds) and which operations are used (`exp` / `dot` → larger multiplier).
- **NaN/Inf consistency**: spurious NaN in the Triton output (where PyTorch has valid numbers) is flagged as a critical bug (likely an unmasked OOB load).
- **Atomic comparison mode**: for atomic ops, element-wise checking is replaced by global-sum and sorted-histogram invariants.

### Reducer

When a failure is detected, the Reducer minimises it through three phases:

1. **Operation pruning** — iteratively removes body ops (in reverse order) while checking that the bug still reproduces.
2. **Dimension reduction** — shrinks `BLOCK_SIZE` and `n_elements` down ladders.
3. **Type simplification** — promotes `float16`/`bfloat16` → `float32` to determine if the bug is precision-specific.

Artifact collection runs a subprocess with diagnostic environment variables (`MLIR_ENABLE_DUMP`, `LLVM_IR_ENABLE_DUMP`, `NVPTX_ENABLE_DUMP` / `AMD_GCN_ENABLE_DUMP`) and captures stdout/stderr. The final package is a zip file containing:

- `reproduce.py` — standalone reproduction script
- `triton_kernel.py` / `torch_ref.py` — isolated sources
- `metadata.json` — full metadata + compile options
- `ir_dumps/` — captured compiler IR logs

## Verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | Outputs match within tolerance |
| `FAIL` | Numerical mismatch beyond tolerance |
| `CRASH` | Runtime exception or missing output |
| `TIMEOUT` | Kernel execution exceeded `timeout_seconds` |
| `COMPILE_ERROR` | `triton.compile()` raised an exception |
| `NAN_MISMATCH` | Inconsistent NaN/Inf propagation |

## License

[MIT](LICENSE)
