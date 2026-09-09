# E023 / T30 — real-Qwen groupwise W4A8 top-32 residual shadow

Status: designed; implementation and guarded run pending.

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
