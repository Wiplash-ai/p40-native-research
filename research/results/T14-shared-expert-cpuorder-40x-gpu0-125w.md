# T14 — exact-Q8 shared-expert 40-layer control

Status: pass.

The control used the real dense-Q8 shared-MLP shapes from every Qwen layer:
two `2048 -> 512` projections, CPU SiLU product, then `512 -> 2048`. It held
GPU0 to 125 W and compared the CPU AVX2/FMA path with the exact CPU-order CUDA
helper, including real input/output transfers around all three projections.

| Measure | Result |
| --- | ---: |
| Layers / matrices | 40 / 120 |
| Q8 weight bytes | 125,829,120 |
| CPU median sweep | 10.7883 ms |
| GPU median sweep | 5.57591 ms |
| Transfer-inclusive speedup | 1.9348x |
| First upload + sweep | 45.628 ms |
| Numerical mismatches | 0 |

All gate, up, activation, and down values were bit-identical. The sub-second
work completed between telemetry samples, so byte accounting provides the
residency record. GPU0 sampled at no more than 37 C, every fan stayed
2,000--2,100 RPM, allocations were released, the guard cooled the cards, and
the original 250 W cap was restored. GPU1 stayed idle.

Decision: exact shared-expert integration is justified in a fourth isolated
Colibri source copy. It must retain the 16-output oracle and allow fallback.

Raw evidence: `research/results/raw/T14-shared-expert-cpuorder-63e129e0.*`.
