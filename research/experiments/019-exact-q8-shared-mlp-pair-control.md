# E019 / T25 — exact-Q8 shared MLP gate/up issue/join

Status: rejected after exact-Qwen canary and fixed-64 comparison.

## Evidence

T24 makes DeltaNet 24.38 ms/token, but shared MLP remains 17.12 ms/token
inside a 30.79 ms/token MoE phase. In the actual decode path, `sh_g` and
`sh_u` are distinct 2048-to-512 exact-Q8 projections with the same `xs`; both
currently use the serial CPU-order helper. The shared-expert scalar gate also
depends only on `xs`, while `sh_d` depends on the SiLU(gate) × up product.

## Smallest exact candidate

1. issue `sh_g` and `sh_u` once with the existing exact Q8 pair helper;
2. compute the scalar shared gate on CPU while the pair is pending;
3. join into the existing host buffers; retain the original SiLU product and
   `sh_d` projection unchanged.

The shared block already overlaps async routed-expert work. The pair shares
GPU0 with some routed experts, so it may lose rather than gain wall-clock time;
only an exact-Qwen canary and fixed-64 comparison can decide.

## Gates

- Default-off `COLI_CUDA_SHARED_PAIR=1`, requiring the existing shared cache.
- One pending pair only; failure recomputes serial exact-Q8 gate/up.
- Exact stdout-hash 16-token canary, then the fixed 64-token oracle.
- Evaluate shared, MoE, and total timers; do not call a local shared reduction
  a win if routed-expert contention worsens total decode.
- Preserve T24's 125 W/card, two-cool-idle-preflight, fan, cooldown, and raw
  evidence protocol.

## Result and decision

T25's exact 16-token canary matched its output oracle and activated both pair
markers. T26's fixed 64-token output also exactly matched its oracle, but the
candidate was slower than T24: 65.26 versus 64.04 ms/token total (+1.9%), with
shared MLP 17.47 versus 17.12 ms/token (+2.0%).

Reject this same-stream gate/up pair. The shared block runs while routed MoE
work is already issued to GPU 0, and the observed total regression is
consistent with contention. That mechanism is not asserted as causal without a
separate controlled profiler. Do not add it to the experimental line or
production. The next user-ranked task is an exact-Qwen attention dependency and
timing audit.
