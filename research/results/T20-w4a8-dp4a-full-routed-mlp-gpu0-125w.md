# T20 — Pascal W4A8 DP4A full routed-MoE MLP control

Status: pass as a synthetic approximate primitive; no Qwen integration.

T20 modeled four per-device Qwen experts using packed W4 gate/up
`2048 -> 512`, SiLU product, intermediate Q8 quantization, and packed W4 down
`512 -> 2048`. Both variants retained identical input and final-output transfer
boundaries; W4A8 used explicit Pascal DP4A while W4A32 used FP32 activations.

| Measure | Result |
| --- | ---: |
| Input/gate/up/hidden/down integer checks | Exact |
| Final relative L2 vs W4A32 | 0.0213725 |
| W4A8 DP4A median | 0.179808 ms |
| W4A32 median | 0.304594 ms |
| Transfer-inclusive speedup | 1.694x |

This passes the declared 5% synthetic error and 1.15x speed gates. Its
maximum relative final-output deviation was 0.352025, so it cannot support an
unqualified model-quality claim; real Qwen weights and generated output still
need a separately designed quality canary.

GPU0 and GPU1 both peaked at 36 C. All fans held 2,000--2,100 RPM. The guard
released allocations, completed cooldown, and restored GPU0 to 250 W.

Raw evidence: `research/results/raw/T20-w4a8-dp4a-mlp-402ac3b2-3d58-40ae-ad2b-6b8750e3e3d2.*`.
