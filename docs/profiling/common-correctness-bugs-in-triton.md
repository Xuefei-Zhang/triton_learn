# Common Correctness Bugs in Triton

- wrong launch grid
- incorrect pointer arithmetic
- missing or incorrect masks near tails
- silently assuming contiguity when the tensor is strided
- wrong dtype assumptions or accumulator precision
- comparing against an untrustworthy baseline

The general debugging order should be:

1. simplify the kernel
2. compare against the baseline
3. reduce the shape
4. inspect intermediate assumptions
5. only then go deeper into generated code
