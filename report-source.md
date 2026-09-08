# P40-native research discovery report

Audience: Jordan and the Terra/Luna implementation agent. Date:2026-09-07. Scope: read-only installed-source/server discovery, bounded primary-source research, concrete experiment design and implementation handoff. No deployment, benchmark, training or model download was performed.

The strongest near-term opportunity is to measure and move Qwen3.6's expensive CPU stages onto available Pascal GPUs, preserving its existing quantized weights and floating activations as a control. The strongest research direction is to test low-bit projections and lookup kernels around a small stable recurrent state. These are related but distinct questions; improving Qwen's runtime does not establish a new neural architecture.

Installed code confirms that the generic GPU controls and planner are not a trustworthy per-family capability map. The expert-size clamp is exactly16,231,956,480bytes, which already covers every routed expert. Qwen's dense/state/head paths remain on the host and its standard server is serial with one KV slot. Explicit source anchors and all relevant settings are recorded in [the execution analysis](research/colibri_execution.md).

Hardware discovery confirmed both P40s, CUDA12 sm61 code, two CPU sockets,503GiB RAM, GPU locality on NUMA0 and a SATA model drive. Repeated FAN2/FAN3 low-speed alerts make load tests inappropriate before diagnosis. Thresholds, current evidence and the proposed independent watchdog contract are in [hardware.md](research/hardware.md).

The literature supports examining diagonal and matrix-state recurrence, ternary projections, lookup-based sums and conditionally updated memory. It does not support replacing Qwen with a new architecture without training or transferring modern Tensor-Core benchmark rates to P40. Primary-source claims, dates and limitations are in [papers.md](research/papers.md); the [gap matrix](research/gaps.md) distinguishes source facts from performance hypotheses.

The deliverable is the [Terra/Luna execution plan](EXECUTION_PLAN.md), with10 primitive families,5 candidate architecture designs, task dependencies, explicit acceptance/stop gates, and a bounded first coding prompt. The next task is a mock-tested guarded measurement harness. Hardware workload gates remain separate from CPU-light coding tasks.

Kimi K3 is real but beyond current target-tier capacity at its official precision. The larger-model analysis provides capacity/bandwidth formulas and explains why additional GPUs and batching do not establish10–20tok/s. No best measured configuration or post-optimization bottleneck is claimed.
