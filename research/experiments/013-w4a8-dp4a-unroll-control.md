# E013 / T18 — Pascal W4A8 DP4A loop-unroll control

Status: source implementation pending CPU-hidden compilation and guarded
runtime acceptance.

## Hypothesis

The reviewed Pascal llama.cpp patch enables DP4A inner-loop unrolling. T17
already has a fully unrolled, eight-warp W4A8 control. Forcing `#pragma unroll
1` with every other element held fixed will establish whether that unrolling is
actually useful for our small expert projection rather than assuming a result
from a different llama.cpp kernel/layout.

## Test and gate

Compile the same source with only `P40_W4A8_UNROLL=0`. It must preserve exact
integer W4/Q8 parity, its 2% error gate, actual `IDP.4A` SASS, and the existing
GPU0 125 W guard. Compare transfer-inclusive median only to T17. Retain the
unrolled variant unless no-unroll is at least 5% faster; a tiny difference is
noise, not a new default. No model source is in scope.
