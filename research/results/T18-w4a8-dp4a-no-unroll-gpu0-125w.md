# T18 — Pascal W4A8 DP4A no-unroll control

Status: pass as a measurement; rejected as the preferred variant.

T18 changed only the DP4A inner-loop directive from unrolled to
`#pragma unroll 1`. It retained packed W4, per-row Q8 quantization, four
`2048 -> 512` experts, tile eight, seed, transfers, CPU integer reference, and
GPU0 125 W policy.

| Measure | T18 no-unroll | T17 unrolled |
| --- | ---: | ---: |
| Integer parity | Exact | Exact |
| Relative L2 error | 0.00384378 | 0.00384378 |
| W4A8 median | 0.0914453 ms | 0.0609619 ms |
| Relative W4A8 latency | 1.50x slower | 1.00x |

No-unroll remains faster than its separately compiled W4/FP32 control
(0.121039 ms), but it loses clearly to the otherwise same unrolled DP4A
kernel. Retain unrolling.

GPU0 peaked at 36 C; GPU1 stayed idle at 37 C. All fans were 2,000--2,100 RPM,
the allocation was released, cooldown passed, and GPU0 restored to 250 W.

Raw evidence: `research/results/raw/T18-w4a8-dp4a-no-unroll-32a72ef5.*`.
