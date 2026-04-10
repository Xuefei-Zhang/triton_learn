# How to Read a Triton Benchmark

When a benchmark says one path is faster, ask:

1. Are both paths doing the same work?
2. Are shapes and dtypes the same?
3. Was warmup performed?
4. Are you measuring only the kernel or also host overhead?
5. Is the result stable across repeated runs?

For this repository, every benchmark note should answer:

- what was measured
- on what shape/dtype/device
- what baseline was used
- what changed
- what explanation seems most plausible
