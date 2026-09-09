# T19 — Pascal W4A8 DP4A 16-warp CTA-tile control

Status: pass as a measurement; rejected as the preferred variant.

T19 changed only the output CTA tile from eight to sixteen warps. It retained
unrolling, packed W4, per-row Q8 quantization, four `2048 -> 512` experts,
seed, transfers, CPU integer reference, GPU0 125 W policy, and safety guard.

| Measure | T19 Tile16 | T17 Tile8 |
| --- | ---: | ---: |
| Integer parity | Exact | Exact |
| Relative L2 error | 0.00384378 | 0.00384378 |
| W4A8 median | 0.0808166 ms | 0.0609619 ms |
| Relative W4A8 latency | 1.32569x slower | 1.00x |

Tile16 passed all gates but is slower, so Tile8 remains selected. T19's own
W4/FP32 control measured 0.115078 ms (1.42394x internal speedup); this is not
compared against T17's separately compiled W4/FP32 control.

GPU0 peaked at 36 C; GPU1 peaked at 36 C. All fans held 2,000--2,100 RPM. The
guard released allocations, completed cooldown, and restored GPU0 to 250 W.

Raw evidence: `research/results/raw/T19-w4a8-dp4a-tile16-549bcc04-f0ba-43b4-a9f2-326454db1a6c.*`.
