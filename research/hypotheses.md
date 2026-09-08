# Hypotheses and cheapest tests

| ID | Hypothesis and prediction | Smallest test | Falsification / next decision |
|---|---|---|---|
| H01 | Qwen's serial CPU trunk dominates reported low GPU-active time | COLI_TIMERS plus actual issue/take events on16/64-token guarded canary | If GPU execution/sync dominates, optimize that measured stage instead |
| H02 | Output head or dense projections have a profitable resident GPU path without changing activation precision | One representative Q8A32 GEMV including activation/result transfer | Reject that placement if complete stage loses; test recurrence/projections next |
| H03 | Full expert residency is already achievable, and more budget yields no gain | Header accounting then startup actual resident/hit/alloc counters | Model accounting already supports it; check runtime success before treating100% projected as measured |
| H04 | Warp-level row reduction beats256-thread shared reduction at expert widths512/2048 | Same W4A32 inputs, one operator, block vs warp tiles | Reject if small row tiles hurt bandwidth or change numerics beyond accepted tolerance |
| H05 | DP4A saves useful time even after dynamic activation quantization | W4A8/W8A8 full operator vs W4A32/Q8A32 at S1/4/8 | Reject at S1 if quantization/packing eats gain; keep possible batching result separate |
| H06 | Small LUTs save decode weight/unpack cost | Include table build, scales, random codes and identical weights | Reject if table bandwidth/gathers dominate; do not train around an unhelpful LUT |
| H07 | FP32 recurrent state plus quantized projections avoids damaging cumulative error | Matched teacher-forced trajectories and tiny training | Reject precision setting on drift/loss/recall; retain FP32 state as control |
| H08 | NUMA0-local placement helps the GPU hybrid more than blanket interleave | Fixed threads, same prompt, one placement change, numa_maps | Reject if CPU memory bandwidth loss exceeds DMA/locality benefit |
| H09 | One P40 matches/beats two for this expert set | One vs two at same CPU/power/thermal settings; measure per-device work | If two materially faster, retain; adding3/4 still needs separate evidence |
| H10 | Conditional memory banks improve recall per touched-state byte | Tiny bank model vs equal-total and equal-active-state controls | Reject if router overhead/collapse/stale state negates recall gain |
| H11 | Independent agents reuse expert weights sufficiently for batching | State-isolated tiny serving plus measured expert-union distribution | Reject linear scaling assumption if unions/queue dominate; identical prompt reuse is not general evidence |
| H12 | A Kimi-scale model fits/is practical through paging | Capacity arithmetic and actual cold-byte throughput first | Fails current storage/format path; no giant download to prove arithmetic |

All but the source/accounting parts of H03 and H12 remain unmeasured. Specify the prediction, numerical tolerance, sample count and stop rule before running each test. Keep null results.
