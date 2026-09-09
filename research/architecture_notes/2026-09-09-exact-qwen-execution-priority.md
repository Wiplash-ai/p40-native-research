# Exact-Qwen execution-path priority

Date: 2026-09-09

## Evidence rule

Primitive controls are implementation checks, not model-performance evidence.
Every candidate must first preserve the relevant primitive output, then pass an
isolated exact-Qwen 16-token stdout-hash canary, then be compared with the
fixed 64-token Qwen protocol. An approximate path additionally needs explicit
activation/logit-quality metrics before it may generate model text. Production
Colibri stays untouched throughout.

## Ranked work

1. DeltaNet exact-Q8 pair issue/join: QKV and Z share an input; overlap their
   unchanged GPU kernels with CPU B/A and remove the intermediate fences.
2. Apply the same dependency-preserving issue/join analysis to the shared MLP
   and routed MoE path, using actual Qwen activations and exact output checks.
3. Profile attention for eligible independent projections before adding a
   batching mechanism there.
4. Revisit W4A8 only with groupwise scales and outlier handling; the current
   per-row scheme is rejected by T21 on real expert activations.
5. Evaluate fusion, persistent execution, and CUDA Graphs only after the
   remaining exact-Q8 wall-clock bottleneck is re-profiled.
6. Treat speculative decoding and expert parallelism as separate algorithmic
   experiments after single-device execution is characterized.

## T23 implementation boundary

The new DeltaNet path is opt-in and exact. It must use the existing CPU-order
Q8 kernel without changing arithmetic, keep QKV/Z outputs on a dedicated
stream until CPU B/A completes, then join before the recurrent state update.
Failure falls back to the existing serial exact-Q8 path.
