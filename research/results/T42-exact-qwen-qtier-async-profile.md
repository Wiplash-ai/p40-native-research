# T42 — exact Qwen asynchronous expert-tier profile

Status: complete attribution run. Exact output and safety gates passed; this
instrumented run is not a speed-selection result.

| Measure | Device 0 | Device 1 | Interpretation |
| --- | ---: | ---: | --- |
| Resident group calls | 3,117 | 3,109 | Both cards serve the routed expert tier. |
| Expert rows | 12,533 | 12,427 | Routing is balanced to within 0.8%. |
| H2D CUDA-event time | 49.45 ms | 42.63 ms | 0.79 / 0.68 ms per timed decode token. |
| Expert-kernel CUDA-event time | 1,191.78 ms | 1,236.75 ms | 18.92 / 19.63 ms per timed decode token. |
| D2H CUDA-event time | 35.21 ms | 31.20 ms | 0.56 / 0.50 ms per timed decode token. |

The aggregate 2,428.53 ms of kernel events is greater than T42's 2,007.8 ms
MoE wall stage (31.87 ms/token). The cards therefore run useful expert kernels
concurrently; their event times must not be summed when estimating latency.
The slower GPU's 19.63 ms/token is the relevant kernel lower bound.

The existing exact-Qwen timers report 2.79 ms/token issue, 16.76 ms shared-MLP
CPU work between issue and take, and 3.72 ms/token in qt_take. That profile
rate was 12.68 tok/s, but it includes event instrumentation and is not compared
as a performance candidate with the 12.90 tok/s adjacent unchanged control.

Canonical stdout SHA-256 remained
5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f.
Safety passed: the 125 W/card run peaked at 42 C, all fans remained
2,000–2,100 RPM, allocations were released, cooldown completed, and both
power limits were restored to 250 W.

Raw evidence: research/results/raw/T42-qwen-qtier-async-profile-2081697d-d180-429f-8cae-2f6edf5ddb54.{json,stderr.txt,stdout.txt}.
