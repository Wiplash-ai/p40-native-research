# E033 / T41 — exact Qwen DeltaNet recurrence-pass fusion

## Hypothesis

Each recurrent value head currently traverses its 128 × 128 state four times:
scale, key/value accumulation, outer update, and query/output accumulation.
Fusing scale with key/value accumulation and outer update with query/output
accumulation removes two state rereads while retaining the same scalar state
writes and `kk`/`vv` accumulation order. This may reduce the CPU recurrence
time identified in T37.

## One-variable change

An isolated T24 source copy adds default-off `COLI_DN_REC_FUSE=1`. The normal
path remains verbatim. The experimental path performs the same rounded scale,
then the same `kk`/`vv` key/value accumulation using that just-written state;
after delta formation it performs the same rounded update, then the same
`kk`/`vv` output accumulation using that just-written state. No weights,
representation, GPU dispatch, routing, OpenMP configuration, or model input
changes.

## Acceptance gates

- The fixed 64-output canonical stdout SHA-256 must remain
  `5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
- The engine must emit its `[dn-rec-fuse] active` marker.
- The normal dual-preflight, 125 W/card, cooldown, release, and restoration
  protocol must pass.
- If end-to-end rate improves, repeat against an adjacent unchanged baseline
  and then add final-logit bitwise validation before retaining the path. If it
  does not improve or the output differs, reject it immediately.

## Result

The required marker and exact stdout hash passed, but T41 measured 12.72 tok/s
and 64.84 ms/token versus the adjacent unchanged T24 control's 12.90 tok/s and
63.93 ms/token. DeltaNet also regressed from 24.41 to 25.14 ms/token. Reject
the fusion: reducing these state rereads does not overcome its code-generation
or register-pressure cost on this CPU. See
[T41](../results/T41-exact-qwen-deltanet-recurrence-fuse.md).
