# DeltaNet CPU locality audit — 2026-09-10

## Scope

This is a static audit plus one exact-Qwen thread-count control. It is not an
inference optimization claim and does not modify production Colibri.

## Hardware and current execution facts

The host exposes 24 physical / 48 logical CPUs across two NUMA nodes:

| Resource | NUMA node 0 | NUMA node 1 |
| --- | --- | --- |
| Logical CPUs | 0-11, 24-35 | 12-23, 36-47 |
| Physical cores | 12 | 12 |
| Tesla P40s | GPU 0, GPU 1 | none |

Both P40 PCIe functions report NUMA node 0. The direct Qwen3.6 binary used by
the fixed-output protocol does not parse `COLI_NUMA` and contains no NUMA
allocation or binding calls. That generic Colibri setting therefore cannot be
treated as an active Qwen3.6 optimization.

## Static recurrence audit

GCC 13.3 compiled `c/qwen36.c` with `-O3 -march=native -fopenmp` and emitted
its full vectorization report in
`research/results/raw/T38-gcc13-qwen36-vectorization.log` (SHA-256
`693c058ef43380fdf4b2ec68e2635efcb3392c339620551e7653b98f525dc677`). The
DeltaNet recurrence function reports six vectorized loops, including 32-byte
AVX2 vector loops. Some loops are versioned for potential aliasing; an
independent source rewrite needs actual-Qwen evidence before it is justified.
The gated normalization loop does not vectorize because of scalar `sqrtf` and
`expf` calls, but T37 measures the entire gated-normalization portion at only
2.160 ms/token, below recurrence's 5.730 ms/token.

## T38 thread-count control

With the exact 64-output SHA oracle, changing only the OpenMP worker count
from 24 to 32 regressed the full model from 12.88 to 12.62 tok/s. DeltaNet
grew by 8.8%, outweighing a 5.0% MoE reduction. Hyperthreading is rejected for
this whole-model configuration; retain 24 physical-core workers.

## Next falsifiable test

The next exact-Qwen control must apply only OS-level
`numactl --interleave=all` to the process, retaining the T24 binary and every
engine variable. It is explicitly a blanket memory-policy test, not a
`COLI_NUMA` test. It must preserve the output SHA and use the normal guarded
thermal procedure. If it fails or regresses, reject blanket interleaving and
consider a separately designed source-level first-touch/local-allocation test
rather than combining policies.
