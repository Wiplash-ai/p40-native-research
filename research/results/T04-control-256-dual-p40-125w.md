# T04 — 256-output Qwen thermal plateau

Date: 2026-09-08. Status: **pass as a single fixed-profile control**, and the
terminal thermal plateau for the current two-P40, 125 W profile. It is not a
performance-setting comparison and does not establish a new best
configuration.

The guarded run used the experimental Qwen binary, the pinned Kreuzzelg
snapshot, the fixed 15-token public prompt, 256 requested outputs, GPUs 0 and
1, `CUDA_EXPERT_GB=auto`, `COLI_DENSE_I8=1`, `COLI_TIMERS=1`, and
`COLI_CUDA_PROFILE=1`. No performance knob changed from the accepted
64-output control. The guard applied 125 W per GPU, required a minimum
five-minute cooldown plus <=40 C, and restored the original 250 W caps.

| Metric | Result |
| --- | ---: |
| TTFT | 2.59 s |
| Decode steps | 255 |
| Decode rate | 1.56 tok/s |
| Engine output rate | 1.54 tok/s (166.5 s for 256 outputs) |
| Total decode time | 163.283 s |
| DeltaNet | 337.99 ms/token |
| DeltaNet projections subset | 234.7 ms/token |
| LM head | 146.91 ms/token |
| Attention | 88.27 ms/token |
| MoE total | 67.15 ms/token |
| GPU0/GPU1 peak temperature | 55 C / 57 C |
| GPU0/GPU1 peak VRAM | 7,895 MiB / 7,895 MiB |
| GPU0/GPU1 peak sampled utilization | 10% / 11% |

The timing split materially matches the accepted 64-output control: DeltaNet
and the CPU-owned LM head still dominate, while the CUDA expert path is only
67.15 ms/token. Expert residency was complete: 10,240/10,240 experts were
resident, there were zero CPU expert misses and zero LFRU swaps, and the two
devices recorded 42,993 and 43,407 tier hits respectively. The async group
counters recorded 21,549 calls / 86,400 experts, with 645 ms H2D, 8,532 ms
kernel, and 217 ms D2H across the complete decode.

Safety cleanup passed: all fans remained 2,000–2,100 RPM, no fresh
fan/device event was recorded, GPU allocations were released, and both 250 W
limits were restored. The run nevertheless reached 55 C / 57 C and reached
the <=40 C recovery target only at the end of the 15-minute cooldown window.
That is a sound reason **not** to run an otherwise-identical 512-output
control: it would add heat and time without distinguishing a new bottleneck.

The text emitted by the direct fixed-prompt harness continues past the short
task and is not a quality evaluation. This control measures decode timing,
residency, and safety only.

## Decision

Do not change `CUDA_EXPERT_GB`, add a generic Colibri CUDA flag, or continue
the output-length ladder. The next experiment is T05: a bounded correctness
and transfer-cost test for a resident GPU implementation of the measured
DeltaNet Q8-weight/FP32-activation projections. It must first prove exact
FP32-activation parity for one projection and beat its complete CPU operator
including transfers before it is wired into a guarded Qwen decode. DP4A
activation quantization is a later, explicitly approximate branch.

Raw evidence:

- [durable guard result](raw/T04-control-256-dual-p40-125w-b4d266c7.json)
- [engine stdout](raw/T04-control-256-dual-p40-125w-b4d266c7.stdout.txt)
- [engine stderr and timers](raw/T04-control-256-dual-p40-125w-b4d266c7.stderr.txt)
