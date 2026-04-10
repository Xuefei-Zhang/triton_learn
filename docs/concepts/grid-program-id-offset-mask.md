# Grid, Program ID, Offset, and Mask

The first conceptual wall in Triton is understanding how one kernel launch turns into many parallel program instances.

## Mental model

1. You launch a grid.
2. Each program instance gets an id.
3. That id determines which block of data it owns.
4. `tl.arange` builds offsets inside the block.
5. A mask prevents invalid reads/writes near the edge.

This is the core pattern behind vector add, reductions, and tiled kernels. If this model is fuzzy, almost every later Triton kernel will feel magical rather than understandable.
