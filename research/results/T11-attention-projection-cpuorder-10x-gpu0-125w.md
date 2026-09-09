# T11 — Exact-Q8 Qwen full-attention projection sweep

T11 is a synthetic, cache-resistant control for the ten full-attention layers
in the pinned Qwen3.6 model. It caches ten distinct Q, K, V, and O projection
sets on GPU0; CPU Q/K normalization, RoPE, KV storage, attention-score
softmax, value accumulation, and gating are explicitly outside this test.

| Metric | Corrected T11 |
| --- | ---: |
| Layers / Q8 matrices | 10 / 40 |
| Cached Q8 weight bytes | 272,629,760 (260 MiB) |
| CPU median sweep | 11.0685 ms |
| GPU complete median sweep | 3.9619 ms |
| CPU/GPU speedup | 2.79373x |
| First upload + sweep | 42.8771 ms |
| Float mismatches | 0 |

All output float bits were identical to Qwen's AVX2/FMA Q8 reduction order.
The initial T11A run used the same arithmetic but emitted invalidly escaped
diagnostic JSON; it is preserved as inconclusive rather than used here.

Safety: GPU0 stayed at 35 C in sampled telemetry under the 125 W cap; GPU1
remained idle at 36 C. Fans held 2,000–2,100 RPM. The guard released memory,
completed cooldown, and restored the 250 W cap. The sub-second workload
completed between polls, so fixed cache-byte accounting is the residency
evidence.

Raw evidence:

- [corrected guard record](raw/T11-attention-cpuorder-a3e20ae8.json)
- [corrected benchmark JSON](raw/T11-attention-cpuorder-a3e20ae8.stdout.txt)
- [T11A inconclusive guard record](raw/T11A-attention-cpuorder-2a0e9cb0.json)
- [T11A malformed benchmark output](raw/T11A-attention-cpuorder-2a0e9cb0.stdout.txt)
