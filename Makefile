PYTHON ?= python

.PHONY: test lint fmt bench-vector-add profile-vector-add

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .

fmt:
	ruff format .

bench-vector-add:
	$(PYTHON) benchmarks/vector_add_bench.py

profile-vector-add:
	$(PYTHON) profiling/profile_vector_add.py
