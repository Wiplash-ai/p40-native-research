# T05C — cache-aware 30-layer DeltaNet Q8 sweep

Date: 2026-09-08. Status: **rejected on the predeclared numerical gate**.

This control swept 30 distinct Qwen-shaped Q8 projection triplets (90 tensors,
1,006,632,960 Q8 weight bytes / roughly 960 MiB), exceeding the host's 60 MiB
per-socket L3 cache. The CPU fixture used Qwen's AVX2/FMA code shape under the
fixed 24-thread OpenMP policy. GPU0 used the existing generic cached format-1
CUDA API at 125 W; each steady-state measurement includes every activation
upload, host boundary, result download, and synchronization.

| Metric | Result |
| --- | ---: |
| CPU sweep samples | 37.519, 38.785, 39.915 ms |
| CPU median | 38.785 ms |
| GPU complete sweep samples | 18.079, 17.910, 17.910 ms |
| GPU complete median | 17.910 ms |
| Observed ratio | 2.166x |
| First sweep, CUDA initialization + ~960 MiB upload | 162.836 ms |
| Cached tensors | 90 / 1,008,353,280 bytes |
| Maximum absolute error | 9.155e-4 |
| Maximum relative error | 4.286e-4 |
| Numerical gate | **fail** |

The observed speed ratio is not an optimization result: the output exceeded
the fixed abs <= 1e-4 + 1e-4 * abs(reference) gate. The next test is
diagnostic-only. It will separately report the direct qkv, z, and dn_out
operator errors, and distinguish a dn_out GPU reduction error from error
propagated through the explicit host boundary. It must pass before any Qwen
integration work.

## Safety

Two preflights were idle at 33 C / 35 C with 0 MiB allocated. The guard
sampled a maximum of 34 C on GPU0, held the minimum five-minute cooldown,
released all GPU allocations, restored GPU0 from 125 W to 250 W, kept every
fan at 2,000–2,100 RPM, and found no new critical fan event. The failure was
only the fixture's numerical exit status.

## Scope correction

Source inspection after the run confirmed that Qwen's DeltaNet phase includes
two additional FP32 dn_b / dn_a projections. This test isolates its three Q8
dn_qkv / dn_z / dn_out matrices only, so a future numerical pass would still
not establish Qwen end-to-end throughput.

Raw evidence:

- [durable guard record](raw/T05C-dn-sweep-815738a9.json)
- [fixture output](raw/T05C-dn-sweep-815738a9.stdout.txt)
- [CUDA banner](raw/T05C-dn-sweep-815738a9.stderr.txt)
