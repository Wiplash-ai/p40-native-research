# T16 — exact-Q8 shared-MLP 64-output comparison

Status: pass.

T16 changed only the output length from T15's 16 to 64. It retained the same
isolated binary, model snapshot, prompt, two-P40 placement, exact-Q8
DeltaNet/LM-head/attention/shared-MLP caches, resident experts, and temporary
125 W/card guard. The required output hash was the established T04/T13
64-output oracle.

| Measure | T16 | T13 control |
| --- | ---: | ---: |
| Outputs / prompt tokens | 64 / 15 | 64 / 15 |
| Output SHA-256 | `5909...f35f` exact | `5909...f35f` |
| TTFT | 0.89 s | 0.90 s |
| Engine output rate | 11.84 tok/s | 6.07 tok/s |
| Decode total | 70.81 ms/token | 151.14 ms/token |
| Shared MLP | 17.17 ms/token | 70.74 ms/token |
| MoE total | 30.63 ms/token | 91.27 ms/token |

That is a 1.95x output-rate increase and 2.13x lower measured decode time
from the one exact shared-MLP change. The full 64-token response was
byte-identical; every one of 10,240 experts was resident with zero actual CPU
expert misses/swaps and no CUDA fallback.

Sampled peaks were 45 C / 45 C and 8,801 / 7,895 MiB. All fans stayed at
2,000--2,100 RPM. The forced runner released memory, cooled to 38 C / 40 C in
its final record, and restored both original 250 W caps.

Decision: retain the exact-Q8 shared cache. The next smallest R&D test is a
standalone approximate W4A8/DP4A MoE-projection control, with explicit integer
correctness and numerical-error accounting before it can affect Qwen.

Raw evidence: `research/results/raw/T16-qwen-shared-cpuorder-6047e0b6.*`.
