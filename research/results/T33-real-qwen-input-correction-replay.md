# T33 — real-Qwen gate/up input correction replay

T33 replays the T32 WGCAP sidecar with a CPU-only C program. It first
reconstructs captured GPU intermediates using original backend-packed W4
weights and scales, then compares correction rankings on untouched holdout.

The reconstruction floor is below `1e-6` aggregate relative L2 for raw gate/up,
hidden, and exact down output. Therefore the small differences between
selectors are not an artifact-format failure.

At K=16 on holdout, down-norm proxy is 1.650% aggregate relative L2 and bounded
nonlinear effect is 1.648%, versus 1.700% for residual magnitude. At K=32, the
corresponding figures are 1.512%, 1.504%, and 1.552%. All 48 held-out records
are below 5% in this per-expert exact-down screen.

Decision: retain the down-norm proxy as the more practical fixed candidate for
one later GPU shadow/cost experiment. It is a modest accuracy enabler, not a
speed result, language-quality result, or production integration.

Raw replay output SHA-256:
`f0342bf2201686252d9a009964089c40db052d1592bb65b02809120e53efb1c7`.
