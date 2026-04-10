# Tiling, Blocking, and Data Reuse

Tiling is the idea that each program instance should reuse data while it is still in fast storage instead of repeatedly pulling it from slower memory.

For matmul-like kernels, the performance story is mostly about arranging work so blocks of input data are reused effectively. This is why Phase 3 matters so much: it is where Triton starts feeling like performance engineering rather than syntax practice.
