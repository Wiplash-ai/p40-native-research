# T15 — exact-Q8 shared-MLP 16-output integration canary

Status: pass.

T15 added only the opt-in exact-Q8 shared MLP cache to T12's isolated binary:
120 `sh_g`, `sh_u`, and `sh_d` matrices for 40 Qwen layers on GPU0. DeltaNet,
LM-head, attention projections, model/prompt, resident expert tier, two-P40
placement, and the 125 W/card forced-command safety policy remained fixed.

| Measure | Result |
| --- | ---: |
| Outputs / prompt tokens | 16 / 15 |
| Output SHA-256 | `43095...0877a` (exact) |
| TTFT | 0.87 s |
| Engine output rate | 8.18 tok/s |
| Decode steps / total | 15 / 71.25 ms/token |
| Shared MLP / router / MoE total | 17.14 / 7.52 / 30.70 ms/token |
| DeltaNet / attention / LM head | 31.92 / 5.29 / 3.35 ms/token |
| T12 short-canary shared phase | 69.86 ms/token |

The required cache marker enumerated all 90 DeltaNet matrices, LM head, 40
attention Q/K/V/O matrices, and 120 shared MLP matrices. All experts remained
resident, with zero CPU expert misses/swaps and no CUDA fallback. The shorter
shared phase is consistent with T14's exact transfer-inclusive operator
result, but this 16-output cold-process canary alone cannot establish a
sustained end-to-end throughput gain.

Telemetry peaked at 42 C / 43 C and 7,537 / 7,517 MiB. Every fan remained
2,000--2,100 RPM. The runner released allocations, reached 37 C / 39 C in its
final record, and restored both original 250 W caps.

Decision: prepare a separately guarded T16 run that changes only output count
from 16 to 64 and requires the previous 64-output response hash.

Raw evidence: `research/results/raw/T15-qwen-shared-cpuorder-8e19ccde.*`.
