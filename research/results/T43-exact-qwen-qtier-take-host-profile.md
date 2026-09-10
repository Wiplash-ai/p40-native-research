# T43 — exact Qwen expert-tier host-take profile

Status: complete attribution run. Exact output and safety gates passed; this
does not select a performance configuration.

| Host take component | Device 0 | Device 1 |
| --- | ---: | ---: |
| Group calls | 3,117 | 3,109 |
| Stream wait | 2.61 ms total; 0.84 microseconds/group | 250.57 ms total; 80.60 microseconds/group |
| Weighted CPU accumulation | 4.11 ms total; 1.32 microseconds/group | 3.98 ms total; 1.28 microseconds/group |

Device 1's residual stream completion accounts for nearly all host-take time.
CPU accumulation is approximately 0.10 ms/model-token when both devices are
considered, so changing its loop order or parallelizing it cannot materially
improve end-to-end latency.

The run preserved canonical stdout SHA-256
5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f.
Its 13.00 tok/s rate is not treated as a winner because this remains an
instrumented, single-run attribution result. Safety passed under 125 W/card:
maximum GPU temperature was 44 C, fans stayed at 2,000–2,100 RPM, allocations
were released, cooldown completed, and both limits returned to 250 W.

Raw evidence: research/results/raw/T43-qwen-qtier-take-host-profile-f25fe21f-832f-4758-af75-c94588e56834.{json,stderr.txt,stdout.txt}.
