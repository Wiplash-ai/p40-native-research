# E006 / T08–T09 — Qwen LM-head exact-Q8 GPU path

Status: T08 standalone and T09 full-model canary pass.

## Source finding

Qwen's largest remaining measured decode phase after T07 is the LM head:
`matmul_d(logit, last, m->lm_head, 1, 2048, 248044)`. It is already converted
to per-output scaled Q8 in the dense-weight registry and currently falls back
to the CPU's exact AVX2/FMA Q8 kernel. This Qwen implementation does not
dispatch its dense path through a `CUDA_DENSE` or `COLI_CUDA_ATTN` environment
flag, so toggling undocumented variables would not test this path.

## T08 standalone result

The isolated exact-order kernel cached a synthetic Q8 LM-head-shaped matrix
(2048 input × 248,044 output; 507,994,112 Q8 bytes) on GPU0. It used the same
32-lane FMA/reduction order as Qwen's CPU Q8 implementation, included input
upload and logits download, and required bit identity across all 248,044
float outputs.

It passed: 16.4813 ms CPU median versus 7.1071 ms cached GPU median (2.319x),
with zero differing float bits. First upload-plus-call cost was 63.6048 ms.
The fixed 125 W guard, fan checks, cooldown, memory release, and power restore
all passed. The rapid workload finished between telemetry samples, so the
fixture's 507,994,112-byte cache accounting—not sampled VRAM—is the residency
evidence.

The synthetic CPU timing is materially below T07's 177.08 ms/token LM-head
timer. Treat T08 only as exact-operator viability: allocation placement,
concurrent model work, and the actual model matrix can change end-to-end
behavior.

## T09 hypothesis and one change

In a new isolated source copy, add an explicit `QDW_LMHEAD_CPUORDER` registry
flag. When `COLI_CUDA_LMHEAD_CPUORDER=1`, upload exactly one already-Q8 LM-head
matrix through the proven helper on GPU0. Preserve the existing 90 DeltaNet
matrices and every other operation. A fixed 16-output Qwen canary must pin the
accepted T04 output hash and reject a missing cache marker, any CUDA diagnostic,
or helper fallback.

## T09 full-model result

T09 passed its fixed 16-output canary. Its 112-byte generated output had the
accepted SHA-256 `43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The required activation line confirmed that GPU0 held the exact-Q8 cache for
all 90 DeltaNet matrices **plus the LM head**. There was no helper fallback or
CUDA runtime diagnostic.

| Decode phase | T09 time |
| --- | ---: |
| DeltaNet | 47.65 ms/token |
| Attention | 123.21 ms/token |
| MoE total | 90.46 ms/token |
| LM head | 3.69 ms/token |
| Total | 265.01 ms/token |

The engine reported 3.20 tok/s (5.0 seconds for 16 output tokens), compared
with 2.07 tok/s for T06C's DeltaNet-only 16-token canary. This is a short
canary, not the sustained comparison. Peak sampled GPU0/GPU1 temperatures
were 46 C / 43 C and peak VRAM was 8,391 / 7,893 MiB. All fans remained
2,000–2,100 RPM; the guarded cooldown, allocation release, and 250 W power
restore passed.

## T10 sustained comparison

T10 passed the separate 64-output identity. It changed only `N_NEW=64` from
T09 and matched the established 419-byte 64-output SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.

| Metric | T04 control | T07 DeltaNet | T10 DeltaNet + LM head |
| --- | ---: | ---: | ---: |
| Engine output rate | 1.52 tok/s | 2.23 tok/s | 3.59 tok/s |
| Decode rate | 1.58 tok/s | 2.29 tok/s | 3.76 tok/s |
| DeltaNet | 330.63 ms/token | 60.36 ms/token | 47.91 ms/token |
| LM head | 146.83 ms/token | 177.08 ms/token | 4.04 ms/token |
| Attention | 85.89 ms/token | 113.58 ms/token | 123.78 ms/token |
| MoE total | 67.57 ms/token | 83.12 ms/token | 90.27 ms/token |
| Total decode step | 630.92 ms/token | 436.3 ms/token | 267.2 ms/token |

T10 is 1.61x faster than T07 and 2.36x faster than the accepted T04 64-output
control. It reduces the model's LM-head time 43.8x versus T07 while retaining
the exact generated result. Its remaining measured bottlenecks are attention
(123.78 ms/token) and MoE (90.27 ms/token), not LM-head arithmetic.

The guard passed at 125 W/card with all fans at 2,000–2,100 RPM. Peak sampled
temperatures were 47 C / 44 C and peak VRAM was 9,343 / 7,895 MiB. It released
allocations, recovered to <=40 C, and restored both 250 W limits. Next is
static source audit of attention and MoE; do not change an undocumented CUDA
flag or run another model workload before a separate, exactness-gated design.
