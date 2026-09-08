# T04 — 64-output Qwen control and cooldown finding

Date: 2026-09-08. The first run failed the thermal-recovery gate despite
successful generation and cleanup. The identical repeat below passed under the
corrected policy and is the valid 64-output control.

The fixed two-P40 125 W profile generated 64 outputs without a runtime abort.
The engine reported 1.50 output tok/s (42.6 s for 64 outputs); its 63 decoded
steps took 39.954 s, or 1.58 decode tok/s. Phase proportions stayed consistent
with the 16-token canary: DeltaNet 333.38 ms/token, LM head 147.48,
attention 86.72, and MoE 66.62. Actual async expert counters again fired:
H2D 183 ms, kernels 2,455 ms, D2H 63 ms across 6,226 groups/24,960 experts.

Safety telemetry was healthy during engine execution: all fans were
2,000–2,100 RPM, no fresh critical SEL appeared, both cards reached 7,895 MiB
expert residency, and GPU0/GPU1 sampled peaks were 47°C/49°C. Both GPU
processes and allocations were released, and both original 250 W limits were
restored.

However, the old guard treated five minutes as an upper cooldown deadline.
At its final sample GPU1 was still 42°C, so it correctly returned
`cooldown_timeout`. Live follow-up showed GPU1 reached 40°C only after about
three additional minutes. This exposes a guard-policy defect, not a reason to
relax the 40°C requirement.

The corrected policy now requires **at least** five minutes **and** <=40°C,
with a 15-minute hard ceiling, continuous fan/device checks, and retained
125 W caps until restoration. Repeat this identical 64-output profile once
under that policy before any longer control or optimization setting is run.

## Corrected-policy repeat: pass

Run ID `6e5cc7ff-574a-40d3-bbdf-9a618f150650` repeated the exact same profile
after two stable idle samples. It passed all safety conditions:

| Metric | Pass result |
| --- | ---: |
| TTFT | 2.28 s |
| Decode steps | 63 |
| Decode phase | 39.748 s |
| Decode rate | 1.58 tok/s |
| Engine output rate | 1.52 tok/s (42.2 s for 64 outputs) |
| GPU0/GPU1 sampled peak | 47°C / 48°C |
| GPU0/GPU1 peak expert VRAM | 7,895 MiB / 7,895 MiB |
| Cooldown until pass | about 8.8 min |
| Fresh fan/device error | none |
| Final state | empty; 250 W limits restored |

Phase timings were stable relative to the failed first attempt: DeltaNet 330.63
ms/token, LM head 146.83, attention 85.89, and MoE 67.57. Expert-event totals
were H2D 186 ms, kernel 2,458 ms, and D2H 64 ms over 6,226 groups/24,960
experts. The two 64-output engine rates (1.50 and 1.52 tok/s) differ by 1.3%;
this is useful control stability evidence, not a speed improvement.

The next workload is a distinct 256-output thermal plateau, with unchanged
model, prompt, GPU set, environment, and 125 W limit. Do not introduce a
performance setting change before that plateau is accepted.

Raw evidence:

- [durable guard result](raw/T04-control-64-dual-p40-125w-889f7f18.json)
- [engine stdout](raw/T04-control-64-dual-p40-125w-889f7f18.stdout.txt)
- [engine stderr and timers](raw/T04-control-64-dual-p40-125w-889f7f18.stderr.txt)
- [passing durable result](raw/T04-control-64-dual-p40-125w-6e5cc7ff.json)
- [passing engine stdout](raw/T04-control-64-dual-p40-125w-6e5cc7ff.stdout.txt)
- [passing engine stderr and timers](raw/T04-control-64-dual-p40-125w-6e5cc7ff.stderr.txt)
