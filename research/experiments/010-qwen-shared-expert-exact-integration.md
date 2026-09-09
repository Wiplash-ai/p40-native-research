# E010 / T15 — Qwen exact-Q8 shared-MLP integration

Status: pass for the fixed 16-output guarded integration canary; a sustained
64-output comparison has not yet run.

## Hypothesis

The accepted T14 control transfers each Qwen-shaped shared MLP layer in 5.58
ms versus 10.79 ms on CPU, while preserving every float bit. Marking the 120
shared MLP Q8 matrices in an isolated Qwen source copy should reduce the
70.74 ms/token shared phase observed in T13, with the established 16-output
response unchanged.

## One change

`colibri-t15-shared` is derived from the T12 isolated source, not the
production engine. Its opt-in `COLI_CUDA_SHARED_CPUORDER=1` flag marks only
`sh_g`, `sh_u`, and `sh_d` in every one of 40 layers for the existing exact
Q8 CPU-order helper. Cache preparation must find exactly 120 shared matrices,
alongside the already enabled 90 DeltaNet, one LM-head, and 40 attention
matrices.

The change leaves routed-expert W4A32 execution, expert routing/tiering,
attention math, CPU SiLU/gating, GPU count, prompt, model, and thermal policy
unchanged. It is neither a W4A8/DP4A experiment nor an approximate model
change.

## Static acceptance

The isolated source compiled with `CUDA_VISIBLE_DEVICES=""`, `CUDA_ARCH=sm_61`,
and the dense-batch, KV-context, JSON-escape, and cache-index regressions
passed. The pinned experimental binary SHA-256 is
`30c9f07bf029209cf7c2931c0b351e89bf15b7411a7fab6b7c2d21cd1c8ca9d8`.

## Guarded canary protocol

The only eligible runtime test is a fixed 16-output dual-P40 canary under a
new restricted forced-command identity. It uses the established prompt/model,
125 W/card cap, watchdog, BMC fan checks, allocation release, cooldown, and
power restoration. It accepts only the previous exact stdout SHA-256
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a` and a
new cache marker explicitly naming all 120 shared matrices. A preflight must
show two idle/cool samples at least 60 seconds apart before it can run.

## Guarded canary result

The first eligible run (`run_id`
`8e19ccde-0eb4-4257-bdf7-253f0e50baff`) passed the exact output oracle and
the required 90+1+40+120 cache marker. It produced the established stdout
SHA-256 `43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`,
with no helper fallback or CUDA diagnostic.

The engine reported 8.18 tok/s (2.0 seconds for 16 outputs; TTFT 0.87 s).
Across 15 decode steps, the shared MLP phase was 17.14 ms/token, down from
69.86 ms/token in the otherwise equivalent T12 short canary. Total decode
time was 71.25 ms/token; DeltaNet was 31.92 ms, attention 5.29 ms, MoE total
30.70 ms, router 7.52 ms, and LM head 3.35 ms. The canary demonstrates that
the exact path works and the intended phase moved; it is not a sustained
throughput claim, because a separate cold process and short output length can
affect other phase measurements.

All 10,240 experts remained resident with zero actual CPU expert misses or
swaps. The guard recorded a 42 C / 43 C peak, 7,537 / 7,517 MiB peak sampled
VRAM, 2,000--2,100 RPM on every fan, allocation release, cooldown to 37 C /
39 C, and restoration of both 250 W caps.

## Decision gate

The canary passed both parity and phase-improvement gates. Run one separate
64-output comparison changing only `N_NEW`, still requiring the established
64-output stdout hash. Retain the path only if it improves total decode time
without a safety failure. Do not combine that measurement with W4A8/DP4A or
MoE-tier changes.
