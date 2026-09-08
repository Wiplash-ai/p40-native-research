# Large-model and multi-P40 direction

The [official Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3) reports2.8T total and104B activated parameters. Its [configuration](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json) has group32 MXFP4 weights with byte scales and higher-precision exclusions. MXFP4 is a floating encoding, not signed INT4 ready for DP4A. Exact checkpoint byte totals remain to be inventoried; no checkpoint download was performed.

## Capacity before throughput

For rounded P=2.8e12, bare weight bytes=P*b/8:

| Hypothetical representation | GiB before workspace/metadata |
|---|---:|
| Uniform16-bit |5215.4|
| Uniform8-bit |2607.7|
| Bare4-bit, no scales |1303.9|
| All weights hypothetically4-bit +1 byte scale/32 |1385.3|
| Bare2-bit |651.9|

These are calculations, not proposed validated quantizations. Mixed high-precision tensors/scales mean the actual official artifact differs from uniform4-bit arithmetic. Model experts are not interchangeable subsets that can be permanently omitted without changing the model.

Actual RAM+two GPUs is nominally503+48=551 GiB before OS/state/copies; four P40s would give599 GiB. The identified AI SSD has about184 GiB free. Even the optimistic combined RAM+VRAM+free-AI-SSD capacity is far short of bare4-bit K3. Other disks exist but available capacity there has not been fully inventoried; this is not a claim about every possible storage location. A supported persistent model backing store must also retain the full checkpoint; RAM cannot replace that requirement for reliable restarts.

Adding two P40s changes residency by48 GiB. That cannot close a700+GiB in-memory bare4-bit gap. At hypothetical2-bit, the whole checkpoint still exceeds RAM+four GPUs before overhead and may be badly degraded. Do not assume arbitrary2-bit/ternary conversion preserves Kimi's quality.

## Throughput limits

For actual per-token traffic D and measured tier bandwidth B:

```
T_token >= max(D_SSD_miss/B_SSD,
               D_RAM/B_RAM,
               max_device(D_PCIe/B_PCIe),
               max_device(D_VRAM/B_VRAM),
               useful_ops/achieved_arithmetic_rate)
```

This is optimistic perfect overlap. Dependent stages and latency add. Use actual active tensor/expert traffic and cache reuse, not total parameters as an unconditional bytes/token value.

As a deliberately optimistic dense-active working-set example,104B parameters at4bits is52 GB/token. Four P40s' aggregate advertised346 GB/s each imply about26.6 tok/s as a bandwidth-only upper ceiling if work is perfectly balanced, every active byte is already in VRAM, and no other work costs time. It is **not** a Kimi prediction: higher-precision modules, scales, repeated reads, state/attention, serial layer dependencies and host/disk misses all lower it. Two-card ceiling for that same toy scenario is about13.3 tok/s.

For an illustrative measured-later SATA rate of0.5 GB/s, one GB of unavoidable fresh expert reads would take at least2seconds. This is a sensitivity example, not a measured drive rate. At10tok/s that bandwidth permits only50MB of cold reads/token; at20tok/s25MB/token. Hence exceptionally high hit/reuse rates would be necessary before other overhead, and routing diversity across agents may lower those rates.

Current Colibrì K3 CUDA path uses device0, uploads experts per call and keeps the nonlinear intermediate on CPU; it is not the envisioned multi-device stationary hot tier. Serving/caching/memory contracts must be re-audited by family. There is currently no evidence supporting10–20tok/s for K3 on this server.

## Multi-P40 experiments worth doing first

- For current Qwen, compare one vs two P40s because all experts fit on one by accounting. Keep measurements of CPU time, per-device imbalance and synchronization.
- For larger models that fit aggregate VRAM, compare layer partitions (small residual exchanges per partition) against expert-local placement (small activations per expert group). Avoid assuming tensor-parallel reductions across PHB are cheap.
- Keep recurrent state with its layer/device. With per-sequence state isolation, batch independent requests to reuse local weights. Each GPU's allocations must fit independently.
- Measure expert union per batch, temporal reuse and misses on unrelated agent prompts; use a shared cache only when hit rates justify it. CPU execution of cold experts can be better than transferring their weights.
- Use CPU/RAM tiers only with measured NUMA placement and bandwidth. Use NVMe-backed ideas after actual NVMe storage exists and has been measured; the current model mount is SATA.

Next sensible size progression is tiny learning models, then fitting7B/13B controls, then30B-class sparse active workloads. A30–100B total MoE may be practical if active bytes and quality fit the measured budget; total parameter count alone is not the target metric.
