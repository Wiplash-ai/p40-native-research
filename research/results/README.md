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

`T29-real-qwen-groupwise-w4a8-outlier-shadow.md` records the fixed top-8
residual correction control. It preserves exact Qwen output but only reduces
the 48 failing groupwise records to 47, so top-8 sparse correction is rejected.

`T30-real-qwen-groupwise-w4a8-outlier32-shadow.md` records the one-variable top-32
control. It improves the distribution but still has 46 failing records; do not
continue increasing K without stage attribution.

`T31-real-qwen-w4a8-stage-attribution-shadow.md` completes that attribution.
It shows that 45 of the remaining 46-tail-scale failures originate before the
gate/up projections; the isolated hidden/down approximation has zero records
above the 5% gate. The next approximation study must focus on gate/up input
residual selection or representation, not another down-side or top-K variant.

`T32-real-qwen-gateup-capture.md` records the guarded, exact-output 96-record
real-Qwen sidecar collection. `T33-real-qwen-input-correction-replay.md` then
validates that sidecar's reconstruction and screens three gate/up input
correction rankings on held-out routed experts. Neither result is a production
change or a performance claim.

`T34-real-qwen-proxy16-shadow.md` closes the most practical T33 candidate on
the full routed Qwen path: it preserves exact output but is rejected because
the 45-record tail remains and its per-group GPU kernels cost 9.72x the
same-stream exact stage at median.

`T35-exact-qwen-deltanet-pair-rebaseline.md` repeats the retained exact-Q8
T24 configuration after the approximation studies. Its byte-identical output,
12.88 tok/s rate, and 64.20 ms/token total show the control remains stable;
it is the pre-instrumentation baseline for the next DeltaNet timing audit.

`T36-exact-qwen-deltanet-pair-profile.md` validates that the accepted pair
already overlaps CPU B/A with QKV/Z GPU work. It rules out CUDA Graphs for
this pair as a material next step and directs the exact-path audit to the
remaining DeltaNet conv, recurrent-state, and norm/output time.

`T37-exact-qwen-deltanet-remainder-profile.md` completes that attribution:
the final output projection is GPU-kernel-bound, while CPU recurrence is now
the largest individual DeltaNet remainder. Its next step is a real-Qwen
CPU-execution experiment rather than another graph or transfer variant.

`T38-exact-qwen-omp32.md` rejects the first real-Qwen recurrence scheduling
control: adding hyperthreads improves MoE but slows DeltaNet enough to worsen
whole-model decode. The retained exact profile stays at 24 workers. The
accompanying [DeltaNet locality audit](../architecture_notes/2026-09-10-deltanet-cpu-locality-audit.md)
records that the direct Qwen binary does not consume `COLI_NUMA`, and that its
recurrence is already AVX2-vectorized; the next guarded control is therefore
OS-level NUMA interleaving, not another ungrounded SIMD or environment sweep.

`T39-exact-qwen-numa-interleave.md` rejects that OS-level control on actual
Qwen: it preserves the canonical output but regresses decode 23.6%, almost
entirely through the DeltaNet recurrence. The next locality work must be a
precise experimental first-touch/allocation study, not a blanket policy.

`T40-exact-qwen-deltanet-firsttouch.md` completes that precise test and
rejects it, too. It proves two-node recurrent-page placement, but the adjacent
unchanged baseline is 36.4% faster. NUMA is no longer an optimization lead for
this exact Qwen path; the next work returns to the recurrence algorithm itself.

`T41-exact-qwen-deltanet-recurrence-fuse.md` rejects that first arithmetic-
order-preserving recurrence rewrite. Its output is exact, but two fewer state
rereads do not improve the full model or the DeltaNet stage. The next lead is
higher-level execution or multi-token work, not another local recurrence loop
rewrite.

T42-exact-qwen-qtier-async-profile.md supplies the missing real-Qwen
multi-GPU evidence. The two P40 resident-expert streams overlap: the combined
38.55 ms-token of expert-kernel CUDA events fits inside a 31.87 ms-token MoE
wall stage, with device 1's 19.63 ms-token kernel time as the relevant GPU
critical-path lower bound. The next attribution pass divides the remaining
3.72 ms-token qt_take host cost into stream wait and CPU accumulation before
considering a synchronization or expert-parallel rewrite.

T43-exact-qwen-qtier-take-host-profile.md shows that the cost is not CPU
combination work. Device 1 accumulates 250.57 ms of wait across 3,109 groups,
while both weighted output accumulations total only 8.09 ms across 6,226
groups. The next exact-Qwen change should target device-1 launch or kernel
latency, not host join parallelism.

T44-exact-qwen-qtier-reverse-issue.md falsifies the smallest device-1 launch
skew control. Reversing only the two existing expert-group issue calls retains
byte-identical output but records 12.83 tok/s / 64.17 ms-token versus 12.90
tok/s / 63.93 ms-token for adjacent unchanged T24. The next exact-path audit
should measure shared-MLP dispatch and overlap, not reorder the host join or
the device launches again.

`T45-exact-qwen-shared-expert-timeline.md` retains a useful but invalid
instrumentation attempt: two of 2,520 layer/token pairs had no routed GPU-0
expert, and the first guard incorrectly counted those ineligible windows as
failures. It is not performance attribution.

`T46-exact-qwen-shared-expert-timeline.md` corrects that accounting and
reconciles 2,518 valid windows plus two `no-gpu0` windows with zero failures.
Shared exact-Q8 work extends GPU-0's local expert tail by about 67 microseconds
per valid window, but GPU 1 is independent and was slower in prior exact-Qwen
profiles. Do not sum the interval into total latency or continue shared-MLP
scheduling work; profile the GPU-1 expert-tail distribution first.
