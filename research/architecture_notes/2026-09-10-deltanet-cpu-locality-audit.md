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

## Blanket interleave result

T39 applied only OS-level `numactl --interleave=all` to the retained exact
T24 path. It preserved the exact output SHA but fell from 12.88 to 9.84 tok/s.
DeltaNet's `l2n+rec` timing grew from 6.8 to 22.5 ms/token. Reject blanket
interleaving: its striped page placement is harmful to this recurrent access
pattern.

## Next falsifiable test

Inspect the exact source locations that allocate and zero DeltaNet's recurrent
state. If state allocation is initialized by the main thread before the static
head-parallel recurrence loop, build one isolated experimental tree that
first-touches each head's state from its eventual worker. Keep the default OS
policy, 24 workers, and exact output oracle. Measure effective `numa_maps`
plus end-to-end Qwen timing. Do not combine this with thread-count, affinity,
or GPU changes.

## Per-head first-touch result

T40 did exactly that and verified 599 of 960 recurrent head slices on node 0,
361 on node 1, and none unknown. It preserved the fixed Qwen output, but its
adjacent unchanged control was 12.90 tok/s versus T40's 8.21 tok/s. The
first-touch path is rejected. Alongside T39, this removes generic and
per-head NUMA placement from the current exact-Qwen priority list. The next
CPU lead is reducing the recurrence's repeated full-state passes while
preserving its arithmetic order and fixed-output oracle.

## Recurrence-pass fusion result

T41 performed that two-pair fusion and preserved the exact Qwen output. Its
adjacent control remained faster (12.90 versus 12.72 tok/s), while DeltaNet
itself grew from 24.41 to 25.14 ms/token. That rejects the local rewrite as a
Pascal/Qwen optimization. Further progress should move up a level: overlap,
multi-token verification, or serving throughput—not more CPU loop surgery.
