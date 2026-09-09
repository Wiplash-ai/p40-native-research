# E013 / T18 — Pascal W4A8 DP4A loop-unroll control

Status: pass; retain loop unrolling.

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

## Guarded result

T18 passed integer parity, its 0.384% relative-L2 error, `IDP.4A` SASS, and
all safety gates, but was decisively slower: 0.0914453 ms W4A8 median versus
T17's 0.0609619 ms (1.50x slower). It is therefore rejected as the preferred
kernel despite still being 1.32x faster than its in-binary W4/FP32 control.
GPU0 stayed at or below 36 C; GPU1 was idle at 37 C; fans, cleanup, cooldown,
and the 250 W power restore all passed.

Decision: retain the unrolled eight-warp control. The next independent change
is a sixteen-warp CTA tile, which respects P40's 1,024-thread limit and tests
the occupancy analogue of the reviewed 64/96 llama.cpp tile.
