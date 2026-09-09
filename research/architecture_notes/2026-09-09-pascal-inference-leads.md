# Pascal inference leads reviewed 2026-09-09

This note separates transferable implementation ideas from claims that cannot
be compared directly with our Qwen3.6/Colibri measurements.

## 1. pascallama.cpp: immediate kernel-design reference

[pascallama.cpp](https://github.com/madsteph74/pascallama.cpp) is a very small
`llama.cpp` patch set for SM 6.1. It changes the Pascal MMQ output tile from 64
to 96 and turns on loop unrolling in nine `__dp4a` vector-dot kernels. The
design assumption is that approximately 28 KiB of shared memory per block
allows two 96-row blocks per Pascal SM.

That is directly relevant to a future Colibri **W4A8 DP4A expert-GEMV control**:
our current Qwen expert path is packed W4A32, whereas the exact Q8 paths use
FP32 activations and preserve CPU reduction order. We will not copy this patch
into Colibri: its layouts and kernel dispatch are different. Instead, use it
as two falsifiable variants of a standalone, Colibri-shaped W4A8 GEMV:

1. `M=64` vs `M=96` output tile, keeping packed-weight layout and all other
   launch geometry identical.
2. Unrolled vs compiler-default DP4A inner loop.

Each variant needs bit/quantization error accounting, steady-state
transfer-inclusive latency, SASS confirmation of `IDP.4A`, and the existing
125 W guard. This is not the next full-model test: T13 must first establish
that the accepted exact attention integration sustains to 64 output tokens.

## 2. DFlash2: promising later serving algorithm, not a Qwen3.6 switch

The reviewed [DFlash2 Pascal model card](https://huggingface.co/maxwelhelp/llama.cpp-DFlash2-pascal6-optimized)
reports a `llama.cpp` branch for a distinct Qwen3.8 27B dense target and its
trained block-diffusion draft model. Its reported best result uses a specific
draft, greedy decoding, GPU-side argmax, Q8 KV, Flash Attention, and an
adaptive speculative chain. Its published 30.26 tok/s figure is therefore not
comparable to our two-GPU 35B-A3B Colibri model and must not be used as a
performance target or a claimed expected gain.

The transferable lesson is experimental: a good **model-compatible** draft can
raise useful tokens/target pass, and its chain length must be tuned against
acceptance rather than maximized. Colibri currently has no demonstrated
Qwen3.6 DFlash2-compatible draft, and its existing doctor recommendation is
`DRAFT=0`. The appropriate future gate is source-audit plus a 16-output
acceptance microtest only after MoE profiling, followed by a draft-length
sweep if and only if the model format and serving path support it.

## 3. Microsoft BitNet: architecture lane, not direct Qwen acceleration

Microsoft's [BitNet](https://github.com/microsoft/BitNet) documents ternary
W2A8 inference with packed two-bit weights, int8 activations, lookup/layout
transforms, and `__dp4a`. This supports our P40-native architecture hypothesis:
for a model trained for ternary weights, a packed low-bit GEMV can trade dense
floating-point traffic for integer decode plus DP4A.

The released GPU kernel is not portable to the P40 as-is. It compiles for
`compute_80`, uses BF16 tensors, and its supplied benchmarks are on an A100.
Pascal lacks native BF16/Tensor Cores. Still, the 16x32 packing and four-value
decode before DP4A are useful reference material for a clean-room Pascal
microbenchmark. Do not retrofit the fixed Qwen weights into ternary form and
call it exact: that would be a separate approximate-quality experiment.

## Updated experiment order

1. T13: exact, hash-pinned 64-output measurement of the already accepted
   DeltaNet + LM-head + attention paths.
2. T14: add phase-local instrumentation to separate shared-expert CPU work,
   PCIe upload/download, kernel time, and take-side synchronization.
3. T15: standalone Colibri-shaped W4A8 DP4A GEMV. Test 64/96 tile and
   unroll variants one variable at a time, using a bounded quality/error gate.
4. Only if T15 earns its gate: an isolated approximate MoE layer experiment,
   then fixed-output quality/latency comparison. It cannot share a run with
   T13 or attention changes.
5. Treat BitNet as a separate small-model/training lane, where ternary weights
   are trained for rather than post-hoc applied to Qwen.
