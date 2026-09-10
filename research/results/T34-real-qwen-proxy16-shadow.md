# T34 — real-Qwen proxy16 gate/up correction shadow

Status: rejected for both real-path fidelity and cost; exact output and safety
controls passed.

| Measure | T34 proxy16 | T31 top-32 input control |
| --- | ---: | ---: |
| Exact stdout SHA-256 | matched | matched |
| Finite routed-group records | 2,397 / 2,397 | 2,397 / 2,397 |
| Relative L2 median | 1.664% | 1.62% |
| Relative L2 p95 | 2.578% | 2.50% |
| Relative L2 p99 | 7.739% | 7.31% |
| Relative L2 maximum | 11.075% | 12.42% |
| Records above 5% L2 | 45 | 45 |
| Minimum cosine | 0.99407 | 0.99481 |
| Proxy kernel median / p95 | 3.331 / 3.935 ms | n/a |
| Exact-stage kernel median / p95 | 0.352 / 0.577 ms | n/a |
| Proxy/exact median / p95 / max | 9.72x / 17.07x / 30.24x | n/a |
| Peak temperature, GPU0 / GPU1 | 45 C / 46 C | 46 C / 47 C |
| Peak VRAM, GPU0 / GPU1 | 7,895 MiB / 7,895 MiB | 7,895 MiB / 7,895 MiB |

The canonical 16-token output SHA-256 remained
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
All correction-cost values are CUDA-event kernel times only, excluding the
shadow output download. Thus the cost rejection is conservative.

T33's low per-expert exact-down replay tail did not survive full routed-group
execution. The proxy has the same 45 above-5% records as T31, while its
all-column Jacobian scan dominates the exact expert stage. Do not tune or
integrate this selector further; return to exact-path launch/synchronization
work before considering a representation change that can remove the scan.

Raw evidence: `research/results/raw/T34-qwen-proxy16-shadow-2568f6f7-3020-4525-ace0-4831df3ccef6.{json,stderr.txt,stdout.txt}`.
