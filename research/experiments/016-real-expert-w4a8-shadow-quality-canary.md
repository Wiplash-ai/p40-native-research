# E016 / T21 — real Qwen expert W4A8 shadow-quality canary

Status: rejected for approximate Qwen expert execution; exact-output shadow run completed.

## Source-grounded target

Qwen decode calls `coli_cuda_expert_group_issue()` from `qwen36_tier.c` after
grouping resident experts by device. On the P40's decode-sized (`total <= 8`)
`fmt == 2` route, `backend_cuda.cu` uploads FP32 rows, dispatches
`grouped_hidden_w4_dual`, then `grouped_down_w4`, and downloads FP32 rows. The
existing W4A16 and WMMA branches are guarded for compute capability 7+ and are
not candidates for Pascal.

## Smallest safe experiment

Create an isolated `colibri-t21-w4a8-shadow` source copy. A new
`COLI_CUDA_W4A8_DP4A=shadow` branch, default-off, may only operate on the
already-selected homogeneous `fmt == 2`, decode-sized group. It will:

1. Quantize the real FP32 expert input rows to signed Q8 on-device.
2. Run grouped DP4A gate/up, fused SiLU product, hidden Q8, and grouped DP4A
   down into dedicated shadow buffers.
3. Copy shadow output to a dedicated pinned buffer, then run the existing exact
   W4A32 kernels and preserve their output as the only value returned to Qwen.
4. After the existing stream synchronization, emit group-level relative L2,
   max absolute error, and finite-value status to stderr only.

Implementation review caught and corrected a signed-nibble packing error before
hardware use: Colibri W4 nibbles are two's-complement values, not offset
binary. The shadow kernel now maps `0..7` to `0..7` and `8..15` to `-8..-1`
before packing each DP4A operand.

The source copy will own separate device/pinned shadow allocations; it may not
reuse unrelated attention/KV scratch. `shadow` cannot change model text. The
first fixed 16-output guarded canary must therefore retain the T04 exact stdout
oracle, show at least one nonzero shadow record, preserve zero CPU fallback,
and report no non-finite values. It is a calibration observation, not a
quality-pass claim. A later approximate-output canary requires separate source,
guard, prompt suite, and explicit human review.

## Decision gate

If every reported real-expert group stays below 5% relative L2, shadow mode
remains exact-output stable, and no thermal/cleanup condition fails, build a
separate approximate-output canary. Otherwise reject the W4A8 engine path and
keep the current exact 11.84 tok/s build.

## Result

T21 collected 2,397 real routed-expert outputs from the fixed 16-token Qwen
canary. The returned exact path preserved the established stdout SHA-256, and
all shadow outputs were finite. However, 288 groups exceeded the 5% relative
L2 gate (median 3.63%, p95 5.69%, maximum 36.37%). The plain per-row Q8
activation scheme is therefore rejected for approximate Qwen MoE execution.

The guard initially marked the run failed because the source wrote a literal
`\\n` rather than a line terminator; raw stderr was parsed offline to recover
the metrics. That observability defect has been corrected, but there is no
reason to repeat the same heat-producing run: the numerical rejection is
already unambiguous. Top-k/logit agreement and DP4A kernel/transfer timing were
not measured because this candidate failed before approximate output can be
considered.
