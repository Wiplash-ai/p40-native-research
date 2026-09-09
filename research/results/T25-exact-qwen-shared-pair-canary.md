# T25 — exact-Qwen shared-MLP pair canary

Date: 2026-09-09

## Scope

This is a correctness and safety canary only. It uses the accepted experimental
DeltaNet QKV/Z pair engine and adds a default-off
`COLI_CUDA_SHARED_PAIR=1` path. The path uploads the shared hidden vector once,
issues exact-Q8 shared gate and up projections, computes the independent scalar
shared gate on CPU, then joins before the unchanged SiLU product and down
projection.

## Result

The fixed 16-token output SHA-256 was exactly the established canary oracle:
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.
Both the DeltaNet and shared-pair markers appeared, with no helper fallback.

The short decode measured 65.35 ms/token total, including 17.54 ms/token for
shared MLP. This is intentionally not treated as a performance comparison;
only the separately guarded fixed-64 T26 comparison can accept or reject the
path.

## Safety

The guard passed. Peak sampled temperatures were 45 C and 46 C; all eight fans
were 2,000–2,100 RPM. Allocations were released, the cooldown passed, and both
GPU power caps were restored to 250 W.

## Evidence

`research/results/raw/T25-qwen-shared-pair-d37e321c-39e4-4183-960c-d6b38c1e8d72.{json,stderr.txt,stdout.txt}`.
