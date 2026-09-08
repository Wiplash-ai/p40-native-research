# P40-native inference research

Discovery date: 2026-09-07. Target: `jordanculver@192.168.1.194` (`excalibur`).

Start with [the Terra/Luna execution plan](EXECUTION_PLAN.md) and the [delegation brief](DELEGATION_BRIEF.md). The first deliverable is a guarded benchmark suite, followed by measured Colibrì changes and tiny architecture experiments. This repository currently contains discovery evidence and specifications, not an implemented benchmark suite or a measured speedup.

## Findings that change the plan

- Installed Colibrì 1.10.1, commit `12a5c464b5c1f8292d578c62458706bc32d6ac95`, offloads Qwen3.6 routed experts. Dense projections, attention, DeltaNet recurrence, routing, shared expert, and output head remain on the CPU. Several generic CUDA flags do not reach this engine.
- The planner's 16.2 GB hot tier equals the entire routed-expert payload. Increasing the budget cannot create more experts. Both P40s have ample additional capacity for a future dense/state GPU implementation.
- The Qwen3.6 gateway currently permits one KV slot; the C server completes requests serially. Continuous batching is an implementation project.
- The BMC repeatedly reports low fan speeds on FAN2/FAN3. No GPU load tests were run during discovery.
- Both P40s attach to NUMA node 0. The AI SSD is SATA, about 447 GiB total with 184 GiB free, rather than NVMe.

## Reading order

1. [Execution plan](EXECUTION_PLAN.md): small tasks, dependencies, acceptance tests, stopping rules, agent prompts.
2. [Colibrì execution map and knob audit](research/colibri_execution.md).
3. [Hardware and thermal findings](research/hardware.md).
4. [Primitive benchmark specification](research/experiments/001-primitives.md).
5. [Architecture candidates](research/architecture_notes/candidates.md).
6. [Literature and prior work](research/papers.md).
7. [Hypotheses](research/hypotheses.md), [experiment log](research/experiments/LOG.md), and [failed/deferred ideas](research/failed_ideas.md).

Optimization objective: useful language-model computation per byte moved and joule consumed, at a stated quality level. Report prefill, decode, startup, and queue time separately. No proposed architecture is claimed novel; no unmeasured throughput is presented as a result.
