# E012 / T17 — P40 W4A8/DP4A MoE-projection control

Status: pass for the synthetic projection control; no model integration is
authorized.

## Hypothesis

After T16, Qwen's GPU-resident routed MoE phase is the largest named decode
phase (30.63 ms/token). Qwen's current small-batch expert kernel multiplies
packed signed W4 weights by FP32 activations. A Pascal-native W4A8 kernel that
quantizes a representative expert input once and uses `__dp4a` may reduce that
projection's transfer-inclusive latency enough to justify a later approximate
MoE-layer experiment.

## Smallest falsifier

The synthetic control represents four routed experts on one P40—the likely
per-device share of Qwen's top-8 route. Each is a real `2048 -> 512` packed-W4
gate/up projection shape. It uploads four FP32 inputs, quantizes them on GPU
to per-row Q8, executes W4 unpack plus DP4A, and downloads four outputs. Its
baseline uses the same persistent W4 weights, host boundaries, warp-per-row
tile, and output shape but retains FP32 activations.

The DP4A integer accumulators must exactly equal a CPU signed-W4/Q8 reference.
The test also reports relative L2 error against W4/FP32 activation output and
fails above 2%. This is an approximate arithmetic control, not a Qwen text
quality claim. It does not load a model or modify Colibri.

## Pascal design note

The reviewed pascallama.cpp 64/96 MMQ tile is not transplanted blindly:
one-token expert GEMV maps a warp to an output row, so 64 output warps do not
fit in one P40 CTA. The first control uses an occupancy-plausible eight-warp
tile. Tile size and loop unrolling become separate variants only if this
baseline proves correct and clears the 15% latency gate. SASS must confirm
`IDP.4A` before interpreting a runtime result.

## Runtime gate

Compile for `sm_61` with CUDA hidden, validate dry-run JSON, inspect SASS, and
use one GPU0-only forced-command run at 125 W after a cool/idle preflight.
Advance only if integer parity, error, fan/thermal/cleanup, and at least 1.15x
transfer-inclusive W4A32/W4A8 speed all pass. A positive result still does not
authorize a Colibri patch; the next stage would be the actual gate/up/down MLP
with an explicit model-quality plan.

## Guarded result

The CPU-hidden `sm_61` build emitted `IDP.4A.S8.S8` in SASS. The guarded GPU0
run (`run_id` `cfb76b1a-4a07-4a1c-85a0-05c3ae82aabb`) passed exact integer
parity, the numerical gate, and all safety/cleanup gates. Its W4A8 DP4A median
was 0.0609619 ms per four-expert projection versus 0.098694 ms for identical
W4/FP32 boundaries: **1.61895x**. Quantization caused 0.00384378 relative-L2
error (0.384%), with 0.06047 maximum absolute error; this is below the 2%
control threshold but is not a model-quality evaluation.

GPU0 peaked at 37 C; GPU1 remained idle at 37 C. Fans stayed 2,000--2,100 RPM,
the allocation was released, cooldown passed, and GPU0's 250 W cap restored.

## Next variant

Test only the `#pragma unroll` decision next. The new binary forces
`#pragma unroll 1`; shape, packed W4 layout, Q8 quantization, output tile,
input/output transfers, seed, and 125 W guard remain fixed. If it is not
faster, retain the unrolled loop and advance to an independent tile-size test.
