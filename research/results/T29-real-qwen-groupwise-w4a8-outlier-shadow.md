# T29 — real Qwen groupwise W4A8 top-8 residual shadow

Status: rejected for approximate expert execution; production unchanged.

This fixed 16-token real-Qwen control retains T28's groupwise signed-Q8 DP4A
bulk computation, then corrects the eight largest post-quantization residuals
against each output's exact packed-W4 values before the gate/up nonlinearity
and down projection. The sparse correction is shadow-only; the original exact
W4/FP32 path still supplied every Qwen result.

| Measure | T29 top-8 | T28 groupwise |
| --- | ---: | ---: |
| Real routed-expert records | 2,397 | 2,397 |
| Non-finite records | 0 | 0 |
| Relative L2 median | 2.39% | 2.46% |
| Relative L2 p95 | 3.02% | 3.12% |
| Relative L2 p99 | 8.19% | 8.46% |
| Relative L2 maximum | 14.17% | 13.18% |
| Records above 5% L2 gate | 47 | 48 |
| Cosine median / minimum | 0.99971 / 0.99359 | 0.99970 / 0.99316 |
| Max absolute error, median / maximum | 0.00365 / 0.06867 | 0.00378 / 0.07846 |
| Peak temperature, GPU0 / GPU1 | 44 C / 45 C | 41 C / 42 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB |

The fixed Qwen stdout SHA-256 matched exactly:
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The guard parsed all records, passed its five-minute cooldown, released GPU
allocations, and restored both cards from the 125 W test cap to 250 W.

Decision: reject top-8 residual correction. It improves ordinary error values
slightly, but it reduces the failing-record count by only one and worsens the
maximum relative L2. The next isolated numerical test may change only top-K
capacity (8 to 32); it must use the same real-Qwen canary and continue to
return exact W4/FP32 outputs. Do not expand it into a deployed sparse kernel,
approximate-output run, or performance benchmark.

Raw evidence: `research/results/raw/T29-qwen-groupwise-outlier-shadow-d506ad55-6d7b-4104-8197-9412b9cf1ec1.{json,stderr.txt,stdout.txt}`.
