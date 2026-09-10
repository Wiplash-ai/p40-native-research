# T45 — exact Qwen shared/expert timeline profile

Status: instrumentation contract rejected. Production Colibrì remains
unchanged; the model output and thermal protocol both passed.

The fixed 64-token stdout matched the canonical SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The profiler collected 2,518 valid GPU-0 shared/expert windows, but two of the
2,520 decode-layer opportunities had no routed expert resident on GPU 0. There
is no same-device expert completion event for those cases, so the guard's
initial requirement for exactly 2,520 valid comparisons correctly failed.

The partial event values are retained as diagnostic evidence only, not a final
attribution: the 2,518 windows report 1,055.12 ms total dense-stream time and
-163.69 ms total from dense-stream end to GPU-0 expert completion. T46 must
separate no-GPU-0 skips from real event failures before interpreting the
per-window relationship.

Safety passed under 125 W/card: maximum GPU temperatures were 43 C / 44 C,
peak allocation was 7,893 MiB/card, fans stayed at 2,000–2,100 RPM, all model
allocations were released, cooldown completed, and limits returned to 250 W.

Raw evidence: `research/results/raw/T45-qwen-shared-expert-timeline-498a37e5-3217-4891-8c57-9fa8f37552c7.{json,stderr.txt,stdout.txt}`.
