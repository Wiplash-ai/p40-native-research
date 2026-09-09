# T21 — real Qwen routed-expert W4A8 shadow canary

Status: rejected for approximate expert execution; production unchanged.

This is the first test using actual Qwen3.6 routed activations and exact model
expert weights rather than a synthetic control. The isolated source copy ran
the Pascal `__dp4a` W4A8 shadow path, then returned the existing exact W4/FP32
CUDA result. The fixed 16-token stdout SHA-256 matched the established exact
oracle: `43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.

| Measure | Result |
| --- | ---: |
| Real routed-expert records | 2,397 |
| Non-finite records | 0 |
| Relative L2 median | 3.63% |
| Relative L2 p95 | 5.69% |
| Relative L2 p99 | 20.66% |
| Relative L2 maximum | 36.37% |
| Records above 5% L2 gate | 288 |
| Cosine median / minimum | 0.99935 / 0.94746 |
| Max absolute error, median / max | 0.00572 / 0.11861 |
| Peak temperature, GPU0 / GPU1 | 42 C / 44 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB |

The run's guard status is deliberately `fail`, but only because a logging
format typo emitted a literal `\\n`, preventing its strict line parser from
seeing shadow records. The raw data nevertheless contains all 2,397 records;
the source and guard were corrected afterward. The exact-output oracle passed,
so this is not a model-output regression.

Decision: do not build an approximate-output canary for this formulation. The
next worthwhile work is full-path profiling and a lower-error formulation
(for example groupwise activation scales or mixed-precision outlier handling),
not forcing the current per-row W4A8 method into production. Logit top-k,
sampled-token agreement, and shadow kernel/transfer latency are intentionally
unmeasured for this rejected candidate.

Raw evidence: `research/results/raw/T21-shadow-parser-failure-5496c4e2-0d51-4b06-b064-b992d4b592c2.{json,stderr.txt,stdout.txt}`.
