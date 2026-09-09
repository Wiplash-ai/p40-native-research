# Results status

`2026-09-07-planner.json` is the unchanged JSON emitted by a read-only call to the installed `resource_plan.build_plan` for the pinned Qwen3.6 checkpoint, two GPUs, context4096 and one KV slot. It records a projection, not actual allocation, residency or performance. The plan's family-independent claims are analyzed in `../colibri_execution.md`.

`2026-09-07-device-attributes.json` is a read-only CUDA driver-attribute snapshot. It created no CUDA context and launched no workload. It confirms installed-device limits useful for the first primitive suite; it is not a throughput or thermal result.

`2026-09-08-fan-policy.md` records the server-side correction from a 20% quiet fan cap to persistent 100% BMC-zone PWM. It verifies configuration application and idle telemetry only; T00's full thermal acceptance gate remains open.

The guarded primitive controls and fixed Qwen decode controls are now recorded
in the individual result files. The longest accepted Qwen control is the
256-output thermal plateau in `T04-control-256-dual-p40-125w.md`; it is a
baseline, not an optimized configuration. Hardware observations and historical
user measurements are separately labeled in `../hardware.md` and
`../experiments/LOG.md`.

`T05G-dn-full-sweep-cpuorder.md` is an exact, transfer-inclusive synthetic
Q8 DeltaNet-projection sweep. It clears its projection-phase performance and
numerical gates; it is not an end-to-end Qwen result.

`T21-real-qwen-expert-w4a8-shadow.md` records the first real Qwen expert
quality test. It rejects the plain per-row W4A8/DP4A formulation for
approximate execution; the exact production path was not changed.

`T22-exact-qwen-async-expert-profile.md` adds nonzero async Qwen routed-expert
event timing and establishes that its device work is kernel-dominated rather
than PCIe-transfer-bound.

`T23-exact-qwen-deltanet-pair-canary.md` records the exact-output canary for
the DeltaNet QKV/Z pair issue/join path. It clears correctness and safety only;
the fixed 64-token comparison remains the performance gate.

`T24-exact-qwen-deltanet-pair-64.md` is that fixed comparison. It accepts the
exact DeltaNet pair path at 12.92 tok/s, with total decode reduced 9.6% from
the matched T16 baseline; it is experimental-only, not a production change.

`T25-exact-qwen-shared-pair-canary.md` and
`T26-exact-qwen-shared-pair-64.md` record the corresponding shared-MLP
gate/up pair path. It clears the exact-output canary but is rejected by the
fixed-64 comparison because total decode worsens 1.9% from the T24 control.

`T27-exact-qwen-attention-subprofile-64.md` records a real-Qwen, exact-output
Q/K/V/O timing audit. It rejects async attention as low headroom: at most the
0.71 ms/token K+V interval is potentially overlapable, before implementation
overhead.

`T28-real-qwen-groupwise-w4a8-shadow.md` records the actual-Qwen groupwise
activation W4A8 shadow control. It materially improves the rejected per-row
formulation but still has 48 of 2,397 records above the 5% L2 gate, so it is
not an approximate-output or production candidate.
