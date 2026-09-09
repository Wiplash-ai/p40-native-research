# T30 — real Qwen groupwise W4A8 top-32 residual shadow

Status: rejected for approximate expert execution; production unchanged.

T30 changes only T29's sparse residual capacity: 32 instead of 8 selected
residuals per activation row. The groupwise Q8 scales, signed W4 DP4A bulk
arithmetic, residual formulation, real-Qwen canary, model output oracle, and
thermal policy were unchanged. As before, all model values came from the
unchanged exact W4/FP32 path.

| Measure | T30 top-32 | T29 top-8 | T28 groupwise |
| --- | ---: | ---: | ---: |
| Real routed-expert records | 2,397 | 2,397 | 2,397 |
| Non-finite records | 0 | 0 | 0 |
| Relative L2 median | 2.21% | 2.39% | 2.46% |
| Relative L2 p95 | 2.79% | 3.02% | 3.12% |
| Relative L2 p99 | 7.40% | 8.19% | 8.46% |
| Relative L2 maximum | 12.63% | 14.17% | 13.18% |
| Records above 5% L2 gate | 46 | 47 | 48 |
| Cosine median / minimum | 0.99976 / 0.99476 | 0.99971 / 0.99359 | 0.99970 / 0.99316 |
| Max absolute error, median / maximum | 0.00333 / 0.06306 | 0.00365 / 0.06867 | 0.00378 / 0.07846 |
| Peak temperature, GPU0 / GPU1 | 46 C / 47 C | 44 C / 45 C | 41 C / 42 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB |

The exact generated-output SHA-256 remained
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The guard parsed all 2,397 records, completed cooldown, released allocations,
and restored both cards from 125 W to 250 W.

Decision: reject the top-K capacity direction. Increasing from 8 to 32
improves central and tail statistics but reduces the failing-record count by
only one. Do not choose another K. The next test must attribute error by
stage—input activation/groupwise gate-up versus post-SiLU hidden/down—not add
more sparse residual capacity or enter an approximate-output path.

Raw evidence: `research/results/raw/T30-qwen-groupwise-outlier32-shadow-4f339e4d-dcd5-4c52-b3c7-407c1197d72d.{json,stderr.txt,stdout.txt}`.
