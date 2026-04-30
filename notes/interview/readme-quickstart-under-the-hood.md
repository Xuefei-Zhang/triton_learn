# Interview Note: README Quickstart Under the Hood

## What does the quickstart actually do?

The four README quickstart commands form a small engineering loop:

1. install the package and dependencies
2. run correctness tests
3. run static lint checks
4. run a benchmark script

They are not redundant. Each command checks a different layer of the repository.

---

## 1. `pip install -e .[dev,runtime]`

### What it means

- `.` means “install the current repository as a Python project”
- `-e` means editable install
- `[dev,runtime]` means also install the optional dependency groups named `dev` and `runtime`

### Which file controls this?

- `pyproject.toml`

### What does it read from `pyproject.toml`?

- build-system settings (`setuptools`, `wheel`)
- package layout under `src/`
- optional dependencies:
  - `dev`: `pytest`, `ruff`
  - `runtime`: `numpy`, `torch`, `triton`

### Why is `-e` important here?

Because this is a learning repo. You will edit code under `src/triton_learn/` frequently. Editable install means Python sees your latest source code immediately without reinstalling after every change.

---

## 2. `pytest -q`

### What it does

Runs the repository's automated correctness tests.

### Which file controls pytest behavior?

- `pyproject.toml`

### Important pytest settings here

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra"
```

### What those settings mean

- `testpaths = ["tests"]` → pytest looks in `tests/`
- `pythonpath = ["src"]` → test files can import `triton_learn` directly
- `addopts = "-ra"` → show useful summary info such as skipped tests

### What actually gets tested?

Examples:

- `tests/runtime/test_env.py` → runtime detection works
- `tests/baseline/test_reference_ops.py` → PyTorch reference ops are correct
- `tests/modules/test_vector_add_module.py` → provider dispatch behaves correctly
- `tests/kernels/test_vector_add.py` → Triton vector-add works when CUDA and Triton are available
- `tests/integration/test_toy_transformer_block.py` → integration block preserves expected shape behavior

### Why might some tests skip?

The Triton/GPU path is environment-dependent. If CUDA or Triton is unavailable, pytest skips those tests instead of hard-failing the whole repository.

---

## 3. `python -m ruff check .`

### What it does

Runs static linting over the repository.

### Which file controls Ruff?

- `pyproject.toml`

### Important Ruff settings here

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
```

### What these rule groups roughly mean

- `E` → style errors
- `F` → undefined names, unused imports, and similar issues
- `I` → import ordering
- `B` → common bug-prone patterns

### What does `check .` mean?

Lint the repository tree from the current directory downward. In practice, that means Ruff checks files in places like:

- `src/`
- `tests/`
- `benchmarks/`
- `profiling/`

### How is this different from pytest?

- `pytest` checks runtime behavior
- `ruff` checks code quality and static hygiene

Passing one does not guarantee passing the other.

---

## 4. `python benchmarks/vector_add_bench.py`

### What it does

Runs the vector-add benchmark script directly.

### Why use `python ...` instead of `pytest ...`?

Because this file is a benchmark script, not a pytest test module.

It has:

- a `main()` function
- a `__main__` entrypoint

It does **not** have `test_*` functions, so pytest does not collect it as a test.

### Which files does it touch?

Main entry file:

- `benchmarks/vector_add_bench.py`

Imported code paths:

- `src/triton_learn/env.py`
- `src/triton_learn/baseline/reference_ops.py`
- `src/triton_learn/kernels/vector_add.py` (only when CUDA + Triton are available)

### Runtime flow

1. insert `src/` into `sys.path`
2. import `reference_vector_add`
3. import `detect_runtime_capabilities`
4. ask whether CUDA and Triton are available
5. if no CUDA → benchmark only the reference path on CPU
6. if CUDA + Triton exist → benchmark both reference and Triton paths on GPU

### Why does it call `torch.cuda.synchronize()`?

GPU kernels are asynchronous. Without synchronization, timing can be misleading because Python may stop the timer before the GPU work actually finishes.

---

## Minimal command-to-file map

| Command | Primary config / entry file | Main effect |
|---|---|---|
| `pip install -e .[dev,runtime]` | `pyproject.toml` | installs package + extras, exposes `src/triton_learn/` |
| `pytest -q` | `pyproject.toml` + `tests/` | runs correctness tests |
| `python -m ruff check .` | `pyproject.toml` | runs static linting |
| `python benchmarks/vector_add_bench.py` | `benchmarks/vector_add_bench.py` | runs benchmark script |

---

## What should I open after running the quickstart?

Best reading order:

1. `src/triton_learn/env.py`
2. `src/triton_learn/modules/vector_add.py`
3. `src/triton_learn/runtime/providers.py`
4. `src/triton_learn/kernels/vector_add.py`
5. `benchmarks/vector_add_bench.py`

This order moves from environment detection, to dispatch, to Triton kernel syntax, to performance measurement.
