# T28 — real Qwen groupwise W4A8 shadow canary

Status: rejected for approximate expert execution; production unchanged.

This is a fixed 16-token Qwen3.6 run on actual routed expert activations and
the checkpoint's W4 expert weights. The isolated `groupwise-shadow` branch
uses a separate signed-Q8 scale for each 256-value activation group, then
uses Pascal `__dp4a` for its shadow gate/up/down results. The existing exact
W4/FP32 result remains the model value, so the canary's generated output is a
strict regression control rather than evidence of approximate model quality.

| Measure | T28 groupwise | T21 per-row |
| --- | ---: | ---: |
| Real routed-expert records | 2,397 | 2,397 |
| Non-finite records | 0 | 0 |
| Relative L2 median | 2.46% | 3.63% |
| Relative L2 p95 | 3.12% | 5.69% |
| Relative L2 p99 | 8.46% | 20.66% |
| Relative L2 maximum | 13.18% | 36.37% |
| Records above 5% L2 gate | 48 | 288 |
| Cosine median / minimum | 0.99970 / 0.99316 | 0.99935 / 0.94746 |
| Max absolute error, median / maximum | 0.00378 / 0.07846 | 0.00572 / 0.11861 |
| Peak temperature, GPU0 / GPU1 | 41 C / 42 C | 42 C / 44 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB |

The returned stdout SHA-256 exactly matched the fixed Qwen oracle:
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The guard parsed the groupwise records, returned `pass`, completed its
five-minute cooldown, released allocations, and restored both 250 W power
limits.

Decision: groupwise scaling is a real numerical improvement, reducing the
over-5% records by 83.3%, but it fails the declared all-records-below-5%
quality gate. It must not be integrated or timed as a deployable W4A8 kernel.
The next isolated real-Qwen control is top-K activation outlier residual
correction over this groupwise baseline. It must preserve the exact canary and
eliminate the remaining error tail before any approximate-output quality or
performance experiment.

Raw evidence: `research/results/raw/T28-qwen-groupwise-shadow-f7d95342-57b4-4cab-86b6-f5d53221aaeb.{json,stderr.txt,stdout.txt}`.
