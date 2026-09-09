# E005 / T07 — Qwen exact-Q8 DeltaNet 64-output comparison

Status: pass.

## Hypothesis

The accepted T06C exact-order Q8 integration reduces Qwen's sustained
DeltaNet cost enough to improve fixed 64-output end-to-end decode throughput
over the accepted T04 control, without changing output text or destabilizing
the two-GPU expert tier.

## One change

T07 is the accepted T06C path with exactly one model-command difference:
`N_NEW=64` replaces `N_NEW=16`. The experimental backend-selector binary,
model snapshot, prompt, GPUs 0 and 1, Q8 expert cache, 125 W/card guard,
OpenMP settings, and cache behavior stay fixed.

The guard pins the accepted 64-output control's stdout SHA-256:
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
It requires the exact-Q8 cache marker and rejects any non-inventory CUDA line
or helper fallback. A separate restricted SSH identity cannot change command,
model, output length, power setting, or timeout.

## Control and prediction

The accepted T04 64-output control measured 1.52 engine output tok/s, 1.58
decode tok/s, and 330.63 ms/token in DeltaNet under the same 125 W/card safety
policy. T06C's accepted 16-output canary measured 2.07 engine output tok/s
and 45.49 ms/token in DeltaNet.

Prediction: the T07 DeltaNet phase remains far below 330.63 ms/token and the
64-output engine rate exceeds the T04 1.52 tok/s control. The relative
end-to-end gain may be less than the short canary because attention, MoE, and
the LM head remain unchanged.

## Falsification and safety gates

Reject the experiment if any one of these occurs:

1. stdout hash differs from the accepted T04 64-output oracle;
2. the exact-Q8 cache marker is absent;
3. any CUDA runtime diagnostic or helper fallback is logged;
4. guard, thermal, fan, cooldown, allocation-release, or power-restore gates
   fail; or
5. output rate does not beat 1.52 tok/s after the same timing definition.

This test is a single fixed comparison, not an authorization to vary cache
sizes, power, sequence length, model, sparse routing, or another kernel. Its
post-run cooldown must reach <=40 C before any later workload.

## Result

T07 passed the fixed 64-output oracle exactly, retained all 10,240 experts in
VRAM with zero CPU misses/swaps, activated the 90-matrix exact Q8 cache, and
logged no CUDA runtime diagnostic or helper fallback. It measured 2.23 engine
output tok/s (28.7 s for 64 outputs) and 2.29 decode tok/s over 63 steps.
Against T04's 1.52 engine tok/s and 1.58 decode tok/s, that is a 1.47x
end-to-end engine-rate improvement and 1.45x decode-rate improvement.

DeltaNet was 60.36 ms/token versus the T04 control's 330.63 ms/token (5.48x
lower). The unmodified phases now dominate: LM head 177.08 ms/token,
attention 113.58, and MoE 83.12. This validates the exact-Q8 DeltaNet path as
a meaningful P40 speedup, while making the LM head the next source-audit
target.

Safety passed: fans stayed 2,000–2,100 RPM; sampled peaks were 46 C / 45 C;
GPU0/GPU1 peak VRAM was 8,855 / 7,895 MiB; the guard's recovery reached
<=40 C; both cards emptied; and original 250 W power limits were restored.
