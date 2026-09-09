# E010 / T15 — Qwen exact-Q8 shared-MLP integration

Status: static acceptance pass; guarded GPU canary pending a fresh cool/idle
preflight.

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

## Decision gate

Record the canary outcome before choosing a 64-output comparison. A passing
hash demonstrates implementation parity only; retain it only if a later,
separate 64-output measurement improves total decode time without a safety
failure. Do not combine its measurement with W4A8/DP4A or MoE-tier changes.
