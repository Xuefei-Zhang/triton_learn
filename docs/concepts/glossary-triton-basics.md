# Glossary: Triton Basics

- **`@triton.jit`**: Marks a Python function as a Triton kernel.
- **`tl.program_id(axis)`**: Returns the current program instance index for a launch axis.
- **`tl.arange(start, end)`**: Creates a vector of offsets inside one program instance.
- **`tl.load` / `tl.store`**: Memory reads and writes from/to pointers.
- **`tl.constexpr`**: Marks a parameter as compile-time constant.
- **Autotune**: Triton’s mechanism for searching launch/meta-parameter configurations.
- **Pointer arithmetic**: Building addresses by combining base pointers and offsets.
- **Fusion**: Combining multiple logical operations into one kernel to reduce memory traffic.
