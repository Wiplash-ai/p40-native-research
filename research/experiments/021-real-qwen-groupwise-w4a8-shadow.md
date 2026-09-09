# E021 / T28 — real-Qwen groupwise-activation W4A8 shadow

Status: designed from T21 and T27; implementation pending.

## Hypothesis

T21's plain per-row Q8 activation scale is overly sensitive to a single large
activation: it expands the quantization step for the whole 2,048-wide expert
input. Splitting the activation vector into fixed 256-value groups should lower
the error of ordinary values without changing model weights or returned Qwen
outputs.

The model's W4 expert weights already use their container's per-output scales
(`GroupDesc.gs`, `us`, and `ds`). This experiment changes only the activation
quantization scale, not the weight representation.

## Exact shadow formulation

For each real routed input row and activation group `g`, compute

`a_g = max(abs(x_i), i in g) / 127`

and quantize `q_i = round(x_i / a_g)`, clamped to signed Q8. For a W4 output
row with its existing weight scale `s_o`, accumulate each integer group before
rescaling:

`y_o = s_o * sum_g(a_g * sum_(i in g)(q_i * w4_oi))`.

Use this for gate/up (2,048-wide, eight groups) and after the unchanged
shadow SiLU product for down (512-wide, two groups). The computation stays
Pascal-native (`__dp4a`) but is shadow-only: existing exact W4/FP32 kernels
still supply the value returned to Qwen.

## Scope and controls

- `COLI_CUDA_W4A8_DP4A=groupwise-shadow` is default-off and valid only for
  homogeneous decode-sized W4 expert groups on sm_61+.
- Fixed group size is 256, divisible by four and shared by the quantizer and
  DP4A kernels. It is an eight-scale input control, not a claim that 256 is
  optimal.
- Fork the already accepted T21 exact-output shadow engine; its purpose is
  numerical comparison on real routed Qwen activations, not end-to-end speed.
  Do not combine the rejected shared pair or T27 timing instrumentation.
- The 16-token actual-Qwen canary retains T21's exact stdout oracle and emits
  separately parseable groupwise metrics.
- Record group count, finite status, relative L2 median/p95/p99/max, max-abs,
  cosine median/minimum, and count above the unchanged 5% L2 gate.
- Do not infer speed from the shadow path. Its multiple group reductions are
  deliberately more expensive than a deployable kernel.

## Decision gate

This is a numerical control only. It clears only if every real group is finite,
the exact returned Qwen stdout matches, and the error distribution materially
improves from T21. An approximate-output candidate remains forbidden unless a
later user-approved gate specifies target error, logits, top-k, deterministic
argmax, and generated-token agreement.

If groupwise scales still leave a meaningful high-error tail, the next isolated
control is top-K activation outlier residual correction. It must add the
residual against dequantized W4 weights in the shadow result and be benchmarked
on the same actual routed activations; do not combine it with fusion or serving
changes.
