# T13 — exact-Q8 attention 64-output comparison

Status: pass.

## Controlled change

T13 used the T12 isolated source binary and changed only the fixed output
length from 16 to 64. It pinned the public prompt, model snapshot, two P40s,
125 W/card temporary cap, NUMA/OpenMP environment, exact DeltaNet cache, exact
LM-head cache, exact Q/K/V/O attention-projection cache, output SHA-256, cache
marker, watchdog, BMC fan checks, allocation release, cooldown, and power
restoration.

## Result

The output SHA-256 was exactly
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`. The
required marker verified 90 DeltaNet matrices, the LM head, and 40 attention
Q/K/V/O matrices on GPU 0; no fallback or unexpected CUDA diagnostic occurred.

| Measure | Result |
| --- | ---: |
| Outputs / prompt tokens | 64 / 15 |
| TTFT | 0.90 s |
| Engine output rate | 6.07 tok/s |
| End-to-end output time | 10.5 s |
| Decode steps / total | 63 / 151.14 ms/token |
| DeltaNet / attention / MoE / LM head | 47.69 / 8.29 / 91.27 / 3.88 ms/token |
| Original T04 64-output engine rate | 1.52 tok/s |
| T10 DeltaNet + LM-head engine rate | 3.59 tok/s |

That is 3.99x T04 and 1.69x T10 under their fixed, hash-matched profiles. It
does not establish a general user-facing throughput figure: model loading and
expert warmstart were included before the timed output measurement, but each
run is a cold process and cache construction still differs from persistent
serving.

## Safety and residency

Every fan stayed within 2,000--2,100 RPM. Telemetry peaked at 45 C and
9,605 MiB / 7,895 MiB (GPU 0 / GPU 1). All 10,240 experts stayed resident;
the explicit counter reports zero CPU cache misses and zero swaps. The guard
released all allocations, waited to 39 C / 38 C in its final sample, and
restored both 250 W power limits. Its durable record has status `pass`.

## Decision

The exact attention cache is retained. Attention is no longer a meaningful
target; MoE is 60% of decode phase time, and shared-expert work is the largest
named subset. The existing `cpu-miss` timer is not a miss counter, so T14 must
instrument the issue/take path before any W4A8 DP4A kernel change is proposed.

Raw evidence: `research/results/raw/T13-qwen-attention-cpuorder-5741a6dd.*`.
