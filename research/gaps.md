# Evidence gaps and confidence

| Decision claim | Evidence and confidence | Contradiction / limit | Next smallest action |
|---|---|---|---|
| GPU/software identity | SSH driver/toolkit/cubin/source hashes; high | No workload executed to verify binary numerical behavior | Guarded correctness canary |
| Low-RPM cooling issue | Repeated BMC readings and SEL assertions; high that alerts occur | RPM briefly recovers; physical fault vs controller/sensor not diagnosed | Serial IPMI sample and physical fan/header inspection |
| Qwen generic flags do not wire GPU trunk | Installed call paths/getenv search; high | Generic docs/planner suggest otherwise | Family capability test, explicit actual-dispatch log |
|16.2GB equals expert payload | Header sum and independent planner match exactly; high | Actual GPU allocation includes contexts/slack; projected hit not observed hit | Guarded startup counters |
| CPU trunk dominates decode | CPU execution confirmed; performance inference medium | No measured phase shares yet; sampled GPU utilization is not FLOPS | T03/T04 phase profile |
| DP4A/LUT can beat W4A32 | Hardware support and prior work; feasibility high, speed unknown | Added quantization/LUT cost, reduction order and loss | E001 same-shape full-operator comparison |
| New recurrent math learns efficiently | Prior related architectures; candidate result unknown | Different state sizes/training budgets can confound comparison | Tiny overfit, recall test, matched training |
| Current Qwen supports serial1-slot serving | Registry and C loop; high | Python queue/concurrent request plumbing can look like batching | Tiny interleaving/state-isolation prototype |
| K3 does not fit available primary tiers | Published rounded parameter count and live target tiers; high | Complete alternate-disk capacity and exact checkpoint manifest not inventoried | Manifest-only size analysis if future storage scope expands |
|10–20tok/s K3/4GPU | Unsupported performance target | Engine lacks proposed multi-GPU hot path and machine has large capacity gap | Solve capacity then measure cold-byte budget; no extrapolated promise |
| Exact P40 cache sizes/profiler interoperability | Primary generic Pascal resources plus installed tools; partial | Exact GP102 byte capacities and driver/profiler tracing not verified | Device attributes and short supported metric probe after thermals |
| Novelty | None established | Closely related SSM/RWKV/BitNet/LUT/MoM/offload work exists | Only after compelling results: deeper closest-prior-art and multi-scale ablations |

Follow-up narrowed from broad literature discovery to these gaps; further unbounded searching is unlikely to resolve them. Current phase is synthesis and handoff; execution evidence is intentionally pending.
