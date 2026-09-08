# T05F — warp-shuffle exact-order Q8 out kernel

Date: 2026-09-08. Status: **rejected for the isolated projection**.

T05F replaced only T05E's shared-memory final reduction and barrier with a
warp-shuffle tree that preserves the same FMA streams and addition order.

| Metric | Result |
| --- | ---: |
| CPU median | 0.1161 ms/call |
| GPU complete median | 0.1245 ms/call |
| Observed ratio | 0.932x |
| Maximum absolute / relative error | 0 / 0 |
| Numerical gate | bit-identical pass |

The shuffle path is exact but slower on this 8 MiB, cache-friendly
single-projection fixture. The CPU median also changed materially between
T05E and T05F despite identical source and fixed settings, confirming that a
single small matrix is not a stable performance decision point.

## Decision

Kill further isolated dn_out kernel tuning. T05G will run the already exact
CPU-order kernel across all 90 Q8 projections in the 30-layer sweep. The 960
MiB weight set exceeds L3 cache and is the smallest fair performance test. It
must pass exact parity and the 15% keep threshold; otherwise direct Q8
projection offload is rejected.

## Safety

The 125 W GPU0 guard passed: maximum sampled temperature 34 C, every fan at
2,000–2,100 RPM, no new fault, empty VRAM, restored 250 W cap, and a complete
cooldown.

Raw evidence:

- [durable guard record](raw/T05F-shuffle-b41017ab.json)
- [fixture output](raw/T05F-shuffle-b41017ab.stdout.txt)
