# E022 / T29 — real-Qwen groupwise W4A8 top-K residual shadow

Status: designed; implementation and guarded run pending.

## Hypothesis

T28's 256-wide groupwise scales remove most of the per-row Q8 error, but 48
of 2,397 real routed-expert records still exceed 5% relative L2. Those records
may be dominated by a sparse set of post-quantization activation residuals.
Correcting only the eight largest residuals per activation row against the
checkpoint's packed W4 weights should remove most of the remaining tail while
retaining groupwise DP4A as the bulk calculation.

## Exact shadow formulation

For an activation `x`, groupwise Q8 reconstruction `xhat`, and a selected
residual `e_i = x_i - xhat_i`, add the sparse correction

`sum(i in top8) e_i * w_i`

to every gate, up, and down output before its corresponding nonlinear step.
The selector operates once before gate/up and once after the shadow SiLU
product before down. It uses the same signed-nibble W4 interpretation and
per-output scale as the exact W4 path.

The first implementation intentionally uses one GPU thread per row to select
top-8 residuals deterministically. It is a numerical control, not a latency
candidate; a parallel selector is premature until the quality premise holds.

## Controls

- New default-off mode: `COLI_CUDA_W4A8_DP4A=groupwise-outlier-shadow`.
- Fixed 256 activation groups and fixed top-K=8.
- Same Kreuzzelg Qwen3.6 checkpoint, fixed public prompt, two P40s, 16 output
  tokens, 125 W/card guard, exact stdout oracle, and T28 statistics parser.
- The original exact W4/FP32 expert launches still return all Qwen values.
- Do not combine with launch, fusion, attention, serving, or multi-GPU changes.

## Decision gate

Passes the numerical hypothesis only if all shadow records are finite, exact
stdout matches, and every one of the 2,397 real routed-expert records is at or
below 5% relative L2. If it fails, record the distribution and reject this
top-8 formulation rather than expanding K blindly. A follow-up must be a
single new capacity choice or a diagnosis of which stage (gate/up or down)
still owns the tail.

No approximate-output, logit, token-agreement, or speed claim is authorized by
this experiment.
