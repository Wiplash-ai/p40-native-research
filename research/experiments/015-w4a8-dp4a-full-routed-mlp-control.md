# E015 / T20 — Pascal W4A8 DP4A full routed-MoE MLP

Status: CPU-hidden build and static assembly check pass; guarded runtime
acceptance pending.

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
