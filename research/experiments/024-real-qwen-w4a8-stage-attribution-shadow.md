# E024 / T31 — real-Qwen top-32 groupwise stage-attribution shadow

Status: designed; implementation and guarded run pending.

## Question

T28–T30 establish that groupwise Q8 activation scaling helps but neither
top-8 nor top-32 sparse correction clears the high-error tail. This control
does not choose another K. It separates the two quantized stages using the
same real routed Qwen activations and top-32 formulation.

1. **Input side:** approximate groupwise/top-32 gate/up, then exact W4/FP32
   down.
2. **Hidden side:** exact W4/FP32 gate/up, then approximate groupwise/top-32
   down.

Both results compare against the exact Qwen expert output. The existing exact
W4/FP32 path still returns all model values.

## Gate

The run must keep the fixed exact stdout oracle, report exactly two finite
metric records per routed expert group, and complete the normal thermal guard.
It is attribution, not a quality pass: select the next formulation only after
the worse stage is empirically identified.
