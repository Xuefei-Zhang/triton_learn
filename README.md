# Triton Mastery Lab

Triton Mastery Lab is a self-contained learning repository for going from “I barely understand Triton syntax” to “I can design, test, benchmark, and explain Triton kernels in an engineering setting.”

The repository is built around one stable architecture:

- `src/triton_learn/` contains the importable Python package.
- `missions/` contains the phase-by-phase learning path.
- `docs/` contains reference material that the missions link to.
- `tests/`, `benchmarks/`, and `profiling/` provide validation and measurement.

The starter implementation is intentionally biased toward immediate usability:

- a working reference baseline layer built with PyTorch
- a working Triton vector-add kernel wrapper
- runtime capability detection and provider dispatch
- a toy integration module that shows how reference components compose
- mission documents for phases 0 through 6

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[dev,runtime]'
pytest -q
python -m ruff check .
python benchmarks/vector_add_bench.py
```

Notes:

- If you only want to inspect the repository structure and markdown content first, you can delay the `runtime` extra install. But to run the reference operators, tests, and benchmark paths, install `'.[dev,runtime]'`.
- The extras are quoted because `zsh` treats square brackets as glob patterns.
- Upgrading `pip setuptools wheel` first avoids editable-install failures like `invalid command 'bdist_wheel'` in fresh virtual environments.
- GPU/Triton tests automatically skip when CUDA or Triton is unavailable.
- The learning path starts in `missions/phase-0-bootstrap.md`.
- The full spec and plan used to build this starter repository are in:
  - `docs/superpowers/specs/2026-04-10-triton-mastery-lab-design.md`
  - `docs/superpowers/plans/2026-04-10-triton-mastery-lab-starter.md`

## Repository Map

```text
docs/                  Reference docs and design artifacts
missions/              The phase-by-phase learning path
notes/                 Personal synthesis and interview-oriented summaries
src/triton_learn/      Importable package with baseline, kernels, runtime, modules
tests/                 Correctness and integration tests
benchmarks/            Repeatable timing scripts
profiling/             Profiling helpers and interpretation notes
playground/            Disposable experiments
```

## Recommended First Session

1. Read `missions/phase-0-bootstrap.md`.
2. Read `docs/phase-guides/phase-0-bootstrap-and-gpu-foundations.md`.
3. Read `docs/concepts/glossary-gpu-basics.md`.
4. Run `pytest -q` and see what is already working.
5. Open `src/triton_learn/kernels/vector_add.py` and `src/triton_learn/modules/vector_add.py`.
6. Compare the Triton path with the baseline path.

## Scope of the Starter Repository

This starter repository does **not** fully implement every advanced Triton kernel from phases 2–6. Instead, it fully implements the project skeleton, learning system, baseline operators, one working Triton kernel path, and the runtime/testing/benchmarking scaffolding that later phases will extend.

That is deliberate: the goal is to give you a repo you can start using immediately without locking all later learning into pre-solved code.
