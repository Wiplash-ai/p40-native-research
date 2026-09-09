# E019 / T25 — exact-Q8 shared MLP gate/up issue/join

Status: designed from the post-T24 real-Qwen profile; not yet implemented.

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
