# E023 / T30 — real-Qwen groupwise W4A8 top-32 residual shadow

Status: complete — rejected for approximate expert execution; production unchanged.

## Rationale

T29 falsified the hypothesis that eight sparse residual corrections explain
T28's remaining high-error tail: the exact-output real-Qwen result changed
from 48 to only 47 records above 5% L2. This follow-up changes exactly one
mathematical capacity parameter: top-K is 32 instead of 8. The quantizer,
group size, weight representation, residual formula, model, prompt, output
length, hardware, and safety guard remain unchanged.

## Hypothesis and gate

Correcting 32 of 2,048 input values (1.56%) and 32 of 512 hidden values
(6.25%) may reduce the tail materially without changing the DP4A bulk path.
This remains a shadow-only numerical control. It passes only if the exact
stdout matches, all 2,397 records are finite, and every record is at or below
5% relative L2.

If this single capacity change fails, record it as a rejection. Do not select
another K without first diagnosing whether the remaining error is activation
quantization, W4 representation/scaling, or nonlinear error propagation.

## Result

The fixed real-Qwen canary passed exact-output and finite-value controls, but
top-32 still leaves 46 of 2,397 records above 5% relative L2. It improves T29
to 2.21% median, 2.79% p95, 7.40% p99, and 12.63% maximum, but does not clear
the quality gate.

The top-K capacity direction is therefore closed. The next admissible control
must attribute error by stage: top-32 groupwise input correction followed by
an exact down projection, versus an exact gate/up projection followed by
top-32 groupwise down correction. That diagnostic may not become an
approximate-output or performance path.
