# T09 — Exact-Q8 DeltaNet plus LM head Qwen canary

T09 adds one opt-in exact-Q8 LM-head cache to the accepted T06C DeltaNet path.
It uses the pinned Kreuzzelg Qwen3.6 model, fixed public prompt, two P40s,
125 W/card, full expert residency, and 16 output tokens.

| Metric | T04 control | T06C DeltaNet | T09 DeltaNet + LM head |
| --- | ---: | ---: | ---: |
| Engine output rate | 1.26 tok/s | 2.07 tok/s | 3.20 tok/s |
| DeltaNet | 354.79 ms/token | 45.49 ms/token | 47.65 ms/token |
| LM head | 156.84 ms/token | — | 3.69 ms/token |
| Total decode | — | — | 265.01 ms/token |

The fixed 16-output SHA-256 exactly matched the accepted oracle:
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
The engine logged `90 DeltaNet matrices plus LM head on GPU 0`; it logged no
CUDA diagnostic or CPU fallback.

This 16-output result establishes functional integration, not sustained
throughput. T10 is the separate 64-output comparison against T07.

Safety: peak sampled GPU0/GPU1 temperatures were 46 C / 43 C; peak VRAM was
8,391 / 7,893 MiB; fans held 2,000–2,100 RPM. The guard completed cooldown,
released GPU allocations, and restored both 250 W limits.

Raw evidence:

- [guard record](raw/T09-qwen-lmhead-cpuorder-eecbbf39.json)
- [engine output](raw/T09-qwen-lmhead-cpuorder-eecbbf39.stdout.txt)
- [engine timers](raw/T09-qwen-lmhead-cpuorder-eecbbf39.stderr.txt)
