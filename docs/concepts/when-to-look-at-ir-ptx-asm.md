# When to Look at IR, PTX, or ASM

Looking at generated code is useful when:

- a kernel is correct but unexpectedly slow
- two configurations behave very differently
- you want to understand whether Triton lowered your intent the way you expected

It is **not** the first debugging move for a beginner. First confirm:

1. the kernel is correct
2. the benchmark is trustworthy
3. the launch configuration is sensible

Only then does generated code inspection become a high-value step.
