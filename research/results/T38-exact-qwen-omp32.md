# T38 — exact Qwen 32-thread CPU control

Status: rejected; canonical output and safety passed, but the one-variable
32-thread configuration is slower than the retained 24-thread control.

T38 reused the T24/T35 binary, model, GPUs, pair path, prompt, power policy,
and exact output oracle. The sole changed value was `OMP_NUM_THREADS=32`,
testing whether 32 independent DeltaNet heads benefit from eight extra logical
workers on the 24-physical-core dual-socket host.

| Measure | T35 / OMP 24 | T38 / OMP 32 | Difference |
| --- | ---: | ---: | ---: |
| Exact stdout SHA-256 | matched | matched | none |
| Engine rate | 12.88 tok/s | 12.62 tok/s | -2.0% |
| Total decode | 64.20 ms/token | 64.99 ms/token | +1.2% |
| DeltaNet | 24.39 ms/token | 26.53 ms/token | +8.8% |
| MoE total | 31.02 ms/token | 29.46 ms/token | -5.0% |
| Attention | 5.44 ms/token | 5.65 ms/token | +3.9% |
| LM head | 3.35 ms/token | 3.34 ms/token | -0.3% |

The exact stdout SHA-256 remained
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
Although 32 workers reduce the MoE total, its 2.14 ms/token DeltaNet regression
more than erases that gain. Retain `OMP_NUM_THREADS=24`; do not use a
whole-model hyperthread count to optimize recurrence.

Safety controls passed: two idle preflights more than 60 seconds apart showed
empty GPUs and 2,000–2,100 RPM fans. The guarded run peaked at 44 C / 44 C and
9,723 MiB / 7,895 MiB, then released allocations, completed cooldown, and
restored both cards to 250 W.

Raw evidence: `research/results/raw/T38-qwen-omp32-2d2a4bfe-12dd-4914-baec-046fbb32761c.{json,stderr.txt,stdout.txt}`.
