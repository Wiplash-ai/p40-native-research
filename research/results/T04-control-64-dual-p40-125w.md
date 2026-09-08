# T04 — 64-output Qwen control and cooldown finding

Date: 2026-09-08. Status: **failed thermal-recovery gate** despite successful
generation and cleanup. This result must not be promoted as a 64-output
baseline until it is repeated under the corrected guard.

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
under that policy before any 512-output baseline or optimization setting is
run.

Raw evidence:

- [durable guard result](raw/T04-control-64-dual-p40-125w-889f7f18.json)
- [engine stdout](raw/T04-control-64-dual-p40-125w-889f7f18.stdout.txt)
- [engine stderr and timers](raw/T04-control-64-dual-p40-125w-889f7f18.stderr.txt)
