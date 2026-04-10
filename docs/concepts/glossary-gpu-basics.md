# Glossary: GPU Basics

- **Kernel**: A function executed on the GPU across many pieces of data in parallel.
- **Grid**: The total launch shape describing how many program instances run.
- **Program instance**: Triton’s unit of parallel execution, roughly the thing identified by `tl.program_id`.
- **Block / tile**: A chunk of data one program instance operates on.
- **Global memory**: Large GPU memory, slower than on-chip storage.
- **On-chip memory / SRAM intuition**: Small, fast storage close to the compute units.
- **Mask**: A boolean condition used to stop out-of-bounds accesses.
- **Occupancy**: A rough measure of how much hardware parallelism is active.
- **Bandwidth-bound**: Performance limited mainly by memory movement.
- **Compute-bound**: Performance limited mainly by arithmetic throughput.
