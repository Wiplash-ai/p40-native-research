# T04 — first guarded Qwen decode canary

Date: 2026-09-08. Status: **pass** for a 16-output thermal/instrumentation
canary; it is not yet the 512-output baseline or a best-configuration claim.

## Fixed configuration

- Experimental binary SHA-256: `d7e8a284a4f864faddd4df6c16c1fbd3c8176222375fb23725af53c453f69051`.
- Checkpoint: pinned Kreuzzelg Qwen3.6 35B-A3B Colibri i4 snapshot.
- Prompt: public 15-token fixture, SHA-256
  `cfd1184857ea076e79a1f67192c228cd75bb4695b014149cb2b6a76a1384af27`.
- Direct experimental C engine; `N_NEW=16`, `COLI_CUDA=1`, `COLI_GPUS=0,1`,
  `CUDA_EXPERT_GB=auto`, `COLI_DENSE_I8=1`, `COLI_TIMERS=1`, and the new
  `COLI_CUDA_PROFILE=1` instrumentation.
- Both cards capped at 125 W for the run. The guard owns GPU/BMC checks,
  process-group termination, five-minute cooldown, and final power restore.

## Result

The engine produced 16 tokens and passed all guard conditions.

| Metric | Result |
| --- | ---: |
| TTFT | 2.60 s |
| Decode steps | 15 |
| Decode phase total | 10.113 s |
| Decode rate | 1.48 tok/s |
| Engine-reported output rate | 1.26 tok/s (12.7 s for 16 outputs) |
| Resident-weight load | 27.7 s |
| GPU expert warmstart | 48.1 s |
| GPU0/GPU1 sampled peak temperature | 42°C / 43°C |
| GPU0/GPU1 sampled peak VRAM | 7,895 MiB / 7,895 MiB |
| Fresh fan-critical SEL event | none |
| Final state | both empty; 250 W limits restored |

The durable result's 407.1-second wall duration includes the mandatory
five-minute cooldown. Its last telemetry sample predates power restoration;
`cleanup.power_restored=true` and a post-run live read confirmed 250 W limits.

## Phase evidence

Per decode token, reported timers were DeltaNet 354.79 ms, attention 90.85 ms,
MoE total 71.71 ms (shared/router are subsets), and LM head 156.84 ms, for
674.20 ms total. The first real asynchronous expert-event counters fired:
2,397 groups / 9,600 experts with aggregate H2D 62 ms, expert kernels 934 ms,
and D2H 25 ms. Their exact scope is the engine's aggregate group statistic;
they must not be added to overlapping wall-phase timers.

All 10,240 experts were resident across the two cards: GPU0 recorded 4,830
and GPU1 4,770 expert hits, with zero CPU misses after warmstart. Each card
actually used about 7.56 GiB for expert tensors despite a larger theoretical
per-device budget. This is evidence that the prior 16.2 GB planner figure is
not the active decode-tier capacity in this run; its planner semantics still
need source-level reconciliation.

## Interpretation and next test

The present decode bottleneck is the CPU-side DeltaNet path, followed by the
LM head—not expert transfers or expert kernels. Low sampled GPU utilization is
therefore expected and does not establish a GPU fault. This short, cold-start
canary is not comparable to the user's historical 1.50 tok/s result or a
commercial persistent-server rate because its 125 W cap, prompt, process
startup, and cache state differ.

Next, implement a separate fixed 64-output guarded control with the exact
same profile and power/cooldown policy. Run it once before extending to the
512-output repetitions required for T04. Do not change an optimization axis
until that longer thermal/control gate passes.

## Raw evidence

- [durable guard result](raw/T04-Qwen-16-dual-p40-125w-86e8c9c4.json)
- [engine stdout](raw/T04-Qwen-16-dual-p40-125w-86e8c9c4.stdout.txt)
- [engine stderr and timers](raw/T04-Qwen-16-dual-p40-125w-86e8c9c4.stderr.txt)
