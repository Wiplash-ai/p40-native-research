# T31 — real Qwen W4A8 stage-attribution shadow

Status: input-side gate/up quantization is the remaining approximate-error
tail; production unchanged.

T31 separates the existing groupwise Q8 plus top-32 FP32 residual shadow into
two real-Qwen paths. The input path approximates gate/up from the routed input
then runs exact W4/FP32 down. The hidden path runs exact W4/FP32 gate/up then
approximates the post-SiLU input to down. Both are shadow-only; model values
still come from the unchanged exact W4/FP32 expert path.

| Measure | Gate/up input shadow | Hidden/down shadow | T30 combined control |
| --- | ---: | ---: | ---: |
| Real routed-expert records | 2,397 | 2,397 | 2,397 |
| Non-finite records | 0 | 0 | 0 |
| Relative L2 median | 1.62% | 1.37% | 2.21% |
| Relative L2 p95 | 2.50% | 2.05% | 2.79% |
| Relative L2 p99 | 7.31% | 2.42% | 7.40% |
| Relative L2 maximum | 12.42% | 2.87% | 12.63% |
| Records above 5% L2 gate | 45 | 0 | 46 |
| Minimum cosine similarity | 0.99481 | 0.99959 | 0.99476 |
| Peak temperature, GPU0 / GPU1 | 46 C / 47 C | 46 C / 47 C | 46 C / 47 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB |

The exact generated-output SHA-256 was
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The guarded run lasted 406.625 seconds, including its mandatory cooldown. It
released allocations, completed cooldown, and restored both cards from 125 W
to 250 W.

Decision: the high-error tail belongs to groupwise/top-32 quantization before
the gate/up projections, not to post-SiLU down input quantization. Reject more
generic top-K and down-side correction. The next eligible approximation study
is a bounded capture/replay of gate/up input residuals with an influence-aware
selector and a calibration/holdout split; it remains shadow-only until it
clears its numerical and later language-quality gates.

Raw evidence: `research/results/raw/T31-qwen-stage-attribution-c60e6556-80c0-43ea-abd3-0cd975c336ab.{json,stderr.txt,stdout.txt}`.
