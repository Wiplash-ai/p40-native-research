# T35 — exact Qwen DeltaNet-pair rebaseline

Status: accepted as a refreshed control; no engine behavior changed.

This run repeats the retained T24 exact-Q8 DeltaNet QKV/Z pair path before
adding instrumentation. It keeps the canonical 64-output prompt, two P40s,
125 W/card guard, GPU 0 DeltaNet residency, and exact output oracle. Its only
purpose is to establish that the performance comparison remains reproducible
after the rejected approximate-path work.

| Measure | T24 accepted control | T35 rebaseline | Difference |
| --- | ---: | ---: | ---: |
| Exact stdout SHA-256 | matched | matched | none |
| Decode tokens | 63 | 63 | 0 |
| Engine rate | 12.92 tok/s | 12.88 tok/s | -0.3% |
| Total decode | 64.04 ms/token | 64.20 ms/token | +0.2% |
| DeltaNet | 24.38 ms/token | 24.39 ms/token | +0.0% |
| Attention | 5.43 ms/token | 5.44 ms/token | +0.2% |
| MoE total | 30.89 ms/token | 31.02 ms/token | +0.4% |
| LM head | 3.34 ms/token | 3.35 ms/token | +0.3% |

The exact output SHA-256 was
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The output is byte-identical to the accepted T24 control. T35's 0.16
ms/token total difference is within the small-run repeat variation and does
not establish a regression.

Safety controls passed: two idle preflights were at 34 C / 36 C and all eight
fans at 2,000–2,100 RPM. The guarded run peaked at 43 C / 43 C and 9,725 MiB /
7,895 MiB. It released both GPUs, completed the required cooldown, and
restored both power limits to 250 W/card.

Decision: retain T24's exact-Q8 path as the control. Advance only to an
instrumentation-only DeltaNet-pair profile that reports host issue/B+A/join
and on-stream H2D/QKV/Z/D2H intervals. Do not compare an instrumented run's
speed directly to T24; use it to select the next optimization target.

Raw evidence: `research/results/raw/T35-qwen-deltanet-pair-rebaseline-96f4feb4-17ea-4af1-87a6-5d21ce38c33e.{json,stderr.txt,stdout.txt}`.
