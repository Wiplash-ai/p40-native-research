# E001: primitive benchmark suite specification

Status: designed, not implemented or run. Owner: Terra for CUDA/numerical kernels, Luna for manifests, telemetry, fixtures and reporting. Execute T01–T04 in EXECUTION_PLAN.md.

## Hypothesis and falsification

Pascal may deliver more useful inference per transferred byte with packed low-bit weights, INT8 dot products, fused recurrence and lookup-based sums than with native FP16 arithmetic. The hypothesis is false for any primitive whose complete operator (including quantization, layout conversion, scales, reductions and transfers when required) loses to the exact FP32-activation control at the same shape and acceptable quality.

## Ten primitive families, in priority order

| ID | Mathematical work | First shapes / variables | Bottleneck to distinguish |
|---|---|---|---|
| P01 | Device streaming copy, vector add, affine/SiLU elementwise | 64/128/256 MiB arrays, contiguous vs strided; repeated/cold data | DRAM vs L2 cache vs launches. Copy counts read+write; FP32 add counts 12N bytes |
| P02 | FP32 FMA vs packed half2 arithmetic; GEMV and GEMM | Square 256/1024/2048; `(O,I)` 512×2048, 2048×512; S=1,2,4,8,16,32 | Instruction latency vs throughput with independent accumulators; library vs custom |
| P03 | Packed INT8 DP4A dot, GEMV and GEMM | Same shapes; signed extremes, tails and aligned K | Dependency chains, packing cost, register use, integer overflow |
| P04 | W4A32 exact packed GEMV and fused expert MLP | Gate/up 512×2048; down 2048×512; groups of 1/2/4/8 experts, S=1 first | Nibble unpacking, 256-thread reduction overhead, redundant activation reads |
| P05 | W4A8 unpack-to-DP4A and W8A8 GEMV | Same weights as P04; group scales row/32/64/128; S=1/4/8/16 | Can saved arithmetic pay for dynamic activation quantization and down-stage re-quantization? |
| P06 | RMSNorm, softmax and warp/block reduction | Width 128/512/2048/8192; context 128/512/2048 | Launch cost, shared barriers, transcendental throughput; FP32/FP64 reference |
| P07 | Fused diagonal-state and delta-rule update | Diagonal D=384,N=16/32; matrix H=32,K=V=128 (actual Qwen); smaller H=12,K=V=32 | State bytes read/written, tiling, per-head parallelism, FP32 drift |
| P08 | LUT/additive/binary/ternary linear operator | Groups g=2/4/8; 2/4/8-bit codes; per-row vs grouped scales; shared/L1 table placements | Table construction, shared-bank conflicts, gathers and reduction; compare DP4A and W4A32 |
| P09 | Structured block sparsity, routed gather/compute/scatter, low-rank factorization | Blocks16/32/64, densities100/50/25/10%; rank32/64/128; shuffled/skewed routing | Index bytes, imbalance, intermediate writes and tiny kernels; equal useful work |
| P10 | CPU↔GPU and GPU↔GPU transfer plus launch/host orchestration | 8 KiB,64 KiB,1 MiB,64 MiB; pinned/pageable; NUMA0 vs interleave; peer eligibility | Latency vs throughput, PHB contention, process/NUMA overhead |

P02 FP16 comparisons must distinguish native half2 math from half storage with FP32 accumulation. P03 must include cuBLAS integer GEMM if the installed library accepts its valid layout; report unsupported explicitly if it does not. CUDA-event measurements are independent of optional profiler support.

For P08, a LUT is an alternative arithmetic implementation: precompute partial activation sums `L_g[b]=Σ_j sign(b_j)x_j`, gather by stored weight codes, accumulate with bitplane/group scales. Table size grows exponentially in group width. Precompute time and table loads count. A lookup index is not an INT8-valued weight, and DP4A does not accelerate the lookup itself.

Power-of-two/log weights and binary XNOR/popcount are small extensions to P08 after the main controls: shifts only substitute cheaply under a suitable fixed-point representation, and XNOR-popcount's dot identity applies to binary operands, not arbitrary activations. Include scaling and conversion. Learned codebooks, ternary sparsity and low rank alter expressivity; they belong in the quality/training branch, not an exact-format control.

## Implementation contract

Proposed files (not yet present):

```text
benchmarks/
  Makefile
  p40_bench.cu
  kernels/{dense,dp4a,w4,reductions,state,lut,sparse}.cuh
  reference.cpp
  config/{canary,standard}.json
  schema/result.schema.json
  tests/{test_guard,test_metrics,test_results}.py
scripts/
  snapshot.py
  guard.py
  run_matrix.py
  summarize.py
```

Build C++17 CUDA with `/usr/bin/nvcc`, explicit `-arch=sm_61`, bounded compiler jobs, CUDA runtime and optional cuBLAS only. No PyTorch/Triton dependency for first suite, no `--use_fast_math`, no accidental newer architecture default. Record tool/library versions and inspect SASS for the emitted DP4A instruction; compilers spelling an intrinsic are not proof that the benchmark measures it. Runtime dimensions/checksum must prevent dead-code elimination. Export actual device attributes, including L2 size, shared limits, registers, SM count, warp size, async engines and peer eligibility.

Each CLI case chooses one GPU, primitive, shape, seed, duration and memory cap. `--list`/`--dry-run` must not initialize CUDA or launch workloads. No case auto-enables both devices. All benchmark invocations go through the watchdog and lock. Add explicit GPU/cpu-only/injected-telemetry modes so orchestration can be verified while cooling is unresolved.

## Timing and bytes

- Allocate and prepare outside steady-state event timing; also report a separate complete-operator/end-to-end interval including necessary per-call transforms and transfers. Report cold initialization separately.
- Warm for a bounded interval, then multiple ≤1-second chunks with cooldown/telemetry checks. GPU events bracket work on its actual stream; synchronize completion before host timer ends. Measure launch floor with empty and small kernels.
- Report median, p10/p90 or IQR, repetitions and timing resolution. Five primitive repetitions are enough initially; do not auto-repeat unstable/aborted tests indefinitely.
- Sweep S=1 versus S>1 explicitly. A peak square GEMM cannot predict decode GEMV. Include expert grouping separately from sequence batching.
- Rotate sufficiently large buffers for a DRAM test; also label hot-cache cases. Algorithmic bytes are not measured DRAM transactions; label counters only if profiler supports and returns them.
- Record useful output elements/tokens, actual operations, packed/scales/index/state/input/output bytes, scratch and peak allocation. `GB/s=algorithmic_bytes/elapsed_seconds/1e9`; `GiB` uses 2^30. DP4A is four MAC/eight arithmetic operations, not four instructions.
- Fixed power/clock policy across comparisons; record actual clocks/throttle reasons. Integrate sampled power over time to estimate GPU joules; short 1-second power estimates are noisy. Whole-system energy stays unavailable unless a real meter/RAPL source is validated.
- P10 reports H2D and D2H separately, concurrent directions and two-device shared-link contention. Check `cudaDeviceCanAccessPeer` before P2P; host bounce is a separate labeled case. Never assume NVLink or peer support.

## Numerical and failure gates

1. Exact integer paths compare against INT64 host references before timing. Test zeros, extremes, alternating signs, unaligned tails, odd dimensions and group boundaries. Bound every INT32 chunk before overflow; don't rely on overflow cancellation.
2. FP32 paths compare absolute/relative errors against a host double reference. Initially use `abs_err <= 1e-4 + 1e-4*abs(ref)` on bounded random cases, plus normwise error for cancellation; record worst case. If reduction order violates it, investigate before changing tolerance. Bit-identical is a stricter, separately reported property.
3. W4 tests round-trip all 16 codes through container→staging→backend packing; verify per-row and gs64 scale addressing separately. Do not confuse signed nibbles with offset-binary values.
4. DP4A quantized-activation tests separate errors from the original FP32-activation result and errors versus their *already quantized* INT64 reference. The latter should be exact before rescaling.
5. State update tests compare the same sequence for 1/128/2048/8192 steps, resets and chunk boundaries. Include extreme decays and persistent small updates; initial state is FP32.
6. GPU OOM/API failures, nonfinite results, correctness failures, watchdog aborts and unsupported metrics fail or skip explicitly. Output `null` for unavailable values, never zero-as-success.

## Screening decisions

Numbers below are predeclared project screening thresholds, not facts from papers.

- Keep a primitive candidate if complete-operator median improves ≥15% over its relevant control with improvement larger than run dispersion and no unexplained numerical error. Keep a pure memory-capacity win only if it enables a stated workload that otherwise does not fit.
- For approximate math, require subsequent held-out language loss/retrieval tests. A >3% perplexity regression or >2 percentage-point retrieval decline is an initial rejection trigger at matched budget; use confidence intervals and multiple seeds before drawing a scientific conclusion.
- If changing routing, expert masks or activation format changes quality, report a separate Pareto point, not a free speedup.
- Microkernel victory is necessary but insufficient: require the chosen end-to-end Colibrì or tiny trained-model measurement to improve before promoting it.

## First falsification ladder

E001a: P01/P02 establish bandwidth and native-half cost. E001b: P03/P05 determine whether activation quantization pays at S=1. E001c: P04 compares stock block reduction against warp-per-row/tiled reduction while retaining W4A32. E001d: P07 determines state update traffic and whether fusion saves it. E001e: P08 chooses a tiny LUT vs DP4A; stop the LUT branch early if overhead wins. E001f: P09/P10 test routed locality and communication. No tiny model training depends on a failed primitive branch.
