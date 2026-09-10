# T46 — exact Qwen shared/expert timeline accounting

Status: complete attribution. Production Colibrì remains unchanged.

T46 reran T45's exact-Q8 shared-MLP versus routed-expert CUDA-event profile
with one accounting correction only: a layer/token with no GPU-0 routed
expert is now `no-gpu0`, not an event failure. The isolated engine, pinned
checkpoint, routing, weights, 64-token workload, and canonical exact-output
oracle were otherwise unchanged.

The fixed stdout matched SHA-256
`5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f`.
The guard reconciled every decode-layer opportunity: 2,518 valid windows + 2
no-GPU-0 windows = 63 decode tokens x 40 layers, with zero event failures.

| Measurement | Total | Per valid window |
| --- | ---: | ---: |
| Shared exact-Q8 CUDA stream | 1,060.48 ms | 0.421 ms |
| Dense-stream end to GPU-0 expert D2H completion | -168.91 ms | -0.067 ms |

The signed second interval is measured from shared stream end to the same
GPU-0 routed expert group's D2H-completion event. Its negative value means
that the GPU-0 expert result was ready about 67 microseconds before shared
stream completion. This makes shared work a small local GPU-0 tail in these
windows. It cannot be summed into model latency: the routed work on GPU 1
proceeds independently and was slower in T42/T43. T46 therefore does not
establish that shared MLP is the whole-model critical path.

The active profile reports 12.84 tok/s / 64.07 ms-token, with MoE 30.44,
shared 17.04, and qtier take 3.42 ms-token. These are instrumentation-bearing
values, not a speed comparison. T26 already showed that changing the shared
gate/up schedule slowed the exact path, so no shared-MLP code is retained as a
performance candidate.

Safety passed under 125 W/card: GPU 0/GPU 1 peaked at 44 C / 44 C, peak VRAM
was 7,893 MiB/card, fan speed stayed at 2,000–2,100 RPM, allocations were
released, cooldown completed, and the guard restored both cards to 250 W.

Decision: do not optimize shared-MLP dispatch further from this evidence. The
next exact-Qwen profile should measure the per-group GPU-1 expert-tail
distribution and residency/load-balance feasibility. Production code remains
untouched.

Raw evidence: `research/results/raw/T46-qwen-shared-expert-timeline-35f20d59-ba65-4d7f-933f-0ae6667b31a5.{json,stderr.txt,stdout.txt}`.
