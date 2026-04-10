# Pitfalls

- A passing benchmark is meaningless if the outputs are wrong.
- A fast Triton path that only works on one happy-path shape is not yet a useful optimization.
- CPU fallback and reference correctness are features, not signs of weakness.
- If you cannot explain the launch grid, you do not really understand the kernel.
