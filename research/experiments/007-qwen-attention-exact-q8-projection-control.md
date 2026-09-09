# E007 / T11 — Qwen full-attention exact-Q8 projection control

Status: pass after one diagnostic-serialization correction and corrected GPU run.

## Source finding

The pinned model has 40 layers: ten `full_attention` layers at indices 3, 7,
11, ..., 39 and thirty DeltaNet layers. In Qwen's `attention()` function, all
four attention projections use CPU `matmul_d()` / per-row Q8 `matmul_q()`:

| Projection | Input | Output | Matrices |
| --- | ---: | ---: | ---: |
| q | 2048 | 8192 | 10 |
| k | 2048 | 512 | 10 |
| v | 2048 | 512 | 10 |
| o | 4096 | 2048 | 10 |

The projection cache is 272,629,760 Q8-weight bytes (260 MiB) before scales.
The remaining q/k normalization, partial RoPE, CPU-resident KV writes,
per-head dot products, softmax, value accumulation, and output gating are all
CPU code. `qwen36.c` does not call `coli_cuda_attention_*`; those backend
entry points implement a different absorption-style attention interface.

## T11 hypothesis

The existing exact CPU-order Pascal Q8 matvec kernel can cache all 40 Q/K/V/O
weights on GPU0 and accelerate the cache-resistant full sweep while preserving
every output float bit. This is an operator test only; it does not claim an
end-to-end attention speedup.

## Fixed test

Synthetic deterministic weights and activations; ten distinct attention-layer
projection sets; CPU AVX2/FMA Q8 reference; cached GPU0 exact-Q8 kernels;
input and output transfer included; all output floats must be bit-identical.
The guard will allow only GPU0, 125 W, 1–3 samples, 288 MiB host accounting,
and the existing fan/BMC/cooldown policy.

Keep only if it is bit-exact and clears the project's 15% control speed gate.
Only then design a separate opt-in Qwen integration canary. Do not toggle
`COLI_CUDA_ATTN`, alter expert placement, or move KV/softmax in this test.

The initial GPU invocation completed internally with an exact result, but its
diagnostic line contained literal escape characters and was not valid JSON.
It is recorded as inconclusive in `T11A-attention-cpuorder-2a0e9cb0.*` rather
than promoted. The correction changes serialization only; it does not affect
arithmetic, allocation, or the fixed workload. The corrected isolated `sm_61`
binary SHA-256 is
`b7d74121b78bf7a7003c83eea88dcacf68754e5de27f122b470e4e83f9427ca9`.
Its fixed guard dry-run reports `cuda_initialized: false` and the output is
validated with `jq` before the corrected guarded run.

## Corrected T11 result

The corrected single-GPU control passed. All 40 matrices and all output float
bits matched the CPU AVX2/FMA reference (`mismatch_count: 0`). The 260 MiB
Q8 cache yielded 11.0685 ms CPU median versus 3.9619 ms cached-GPU median,
for a 2.79373x speedup. First upload plus sweep took 42.8771 ms.

GPU0 was 35 C throughout sampled telemetry, with 125 W enforced; all eight
fans stayed 2,000–2,100 RPM. The sub-second work finished between polls, so
the fixed 272,629,760-byte Q8 cache accounting is the residency evidence.
The guard completed its cooldown, released allocations, and restored 250 W.

T11 clears the exactness and 15% speed gates. The next eligible work is a
new-source-copy, opt-in Q/K/V/O integration canary that preserves CPU KV,
RoPE, softmax, and gating. It must use a new output oracle and change no MoE,
expert-tier, or power setting.
