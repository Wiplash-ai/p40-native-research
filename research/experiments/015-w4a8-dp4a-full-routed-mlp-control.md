# E015 / T20 — Pascal W4A8 DP4A full routed-MoE MLP

Status: pass as a synthetic approximate primitive; no engine integration.

## Hypothesis

The selected T17 eight-warp DP4A projection improves one quantized expert
matrix. A routed Qwen expert is instead a gate and up projection, SiLU product,
hidden re-quantization, then down projection. The intermediate quantization
and nonlinearity may erase its projection-level gain.

## Smallest falsifiable test

T20 is a GPU0-only synthetic four-expert MLP matching Qwen's
`2048 -> 512 -> 2048` geometry and packed signed W4 weights. W4A8 quantizes
the input, runs both gate/up projections with DP4A, dequantizes, applies SiLU
product, quantizes the intermediate, and runs the down DP4A projection. W4A32
uses the identical weight/input/output transfer boundaries and nonlinearity.

The CPU reference independently verifies input-Q8, gate-Q8 accumulator,
up-Q8 accumulator, hidden-Q8, and down-Q8 accumulator equality. The final
W4A8 output must be within 5% relative L2 of the W4A32 control and faster by
at least 15% before any later quality-specific work is considered. This is not
an engine integration authorization.

## Static acceptance

The `sm_61` CPU-hidden build produced SHA-256
`cabeebe044008c931df0f1d08e12c8201bd571c91015cf8bd89ceea3cdc0b42e`.
Its dry-run returned before CUDA initialization, and SASS contains 20
`IDP.4A.S8.S8` instructions. The GPU measurement remains separately gated by
a distinct fixed-command key, fresh cool/idle preflight, GPU0-only 125 W cap,
and cooldown/restore checks.

## Guarded result

T20 passed every staged integer check: input Q8, gate accumulator, up
accumulator, hidden Q8, and down accumulator exactly matched their independent
CPU W4/Q8 references. Final W4A8 output was 2.13725% relative L2 from the
same GPU's W4/FP32 full-MLP control, below the predeclared 5% gate. Its
transfer-inclusive median was 0.179808 ms, versus 0.304594 ms W4/FP32:
1.694x faster. GPU0/GPU1 stayed at or below 36 C, fans at 2,000--2,100 RPM,
and allocation release, cooldown, and 250 W cap restore passed.

Decision: the P40 W4A8/DP4A arithmetic is viable for the complete synthetic
routed MLP, not merely a one-matrix microbenchmark. It still does not establish
Qwen quality on real weights or authorize an engine patch. The next eligible
step is a source-level proposal for a separately isolated, opt-in real-expert
quality canary with no exact-output claim.
