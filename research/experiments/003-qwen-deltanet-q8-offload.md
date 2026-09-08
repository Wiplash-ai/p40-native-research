# E003 / T05 — Qwen DeltaNet Q8-weight GPU offload control

Status: T05A pass; T05B three-projection control is next. T05A used no model
load and no kernel modification.

## Why this is next

The accepted fixed Qwen controls establish that resident routed experts are
not the current decode ceiling: they account for 67.15 ms/token, while
DeltaNet accounts for 337.99 ms/token. The DeltaNet sub-timer attributes
234.7 ms/token to its projections. That is the highest measured CPU-owned
candidate that can be isolated without changing Qwen's mathematics.

This is not a DP4A or W8A8 experiment. On this x86 host, Qwen's current
`matmul_d()` path resolves its registered dense weights to `matmul_q()`:

```
y[o] = dense_scale[o] * sum_i(float32(x[i]) * int8(weight[o,i]))
```

The CUDA backend's existing format-1 path implements the same stored Q8
weights, per-row scales, and FP32 activations. It persists a `ColiCudaTensor`
after the first upload, then transfers only the activation and result for each
call. Reduction order differs, so numerical agreement must be tested; it is
not reasonable to require bit identity.

## Exact target shapes

The pinned checkpoint has hidden width 2048, 40 layers, and a repeating
three-DeltaNet/one-attention topology: 30 DeltaNet layers. Each DeltaNet layer
has three Q8 dense projections:

| Projection | I | O | Q8 weights per layer |
| --- | ---: | ---: | ---: |
| `dn_qkv` | 2048 | 8192 | 16 MiB |
| `dn_z` | 2048 | 4096 | 8 MiB |
| `dn_out` | 4096 | 2048 | 8 MiB |

Together those are 32 MiB/layer, or about 960 MB plus per-row scales for all
30 DeltaNet layers. They fit comfortably in one P40 alongside the accepted
expert cache; first T05 does **not** allocate all of them or use two GPUs.

## Smallest falsifying experiment: T05A

Build an isolated, deterministic `sm_61` fixture against the existing
`coli_cuda_matmul()` API. It owns synthetic host Q8 weights, per-output
scales, and FP32 activations; it does not parse or load a model.

1. Verify CPU reference and GPU output for each target shape with zeros,
   alternating signs, deterministic random data, and an unaligned-tail case.
   Require finite output and the project-wide FP32 gate
   `abs_error <= 1e-4 + 1e-4 * abs(reference)` for every element.
2. Measure the CPU implementation matching Qwen's `matmul_q()` arithmetic,
   then the complete cached GPU call: H2D activation, kernel, D2H result, and
   synchronization. Record first-upload time separately and exclude it only
   from the steady-state comparison because Qwen retains the tensor.
3. Use GPU0 only at 125 W, with one fixed one-second workload after a dry run
   and two stable idle samples. The server guard must retain the existing
   fan/SEL/temperature/watchdog/cleanup policy and restore 250 W. No Qwen
   binary, no model weights, and no GPU1 work belong in T05A.
4. Run five bounded repetitions only if the first run is correct and cool.
   Compare medians and dispersion rather than one timing.

Keep the path only if it is at least 15% faster than the matching CPU complete
operator, beyond run dispersion, and passes every numerical and thermal gate.
Otherwise reject direct per-projection offload and move to the LM-head control
or a different mathematical primitive; do not compensate by silently
quantizing activations.

## T05A result

The guarded GPU0 control passed on 2026-09-08. Its 2048x8192 `dn_qkv`
projection measured 4.645 ms/call on the matching CPU operator and 0.335
ms/call through the cached, transfer-inclusive GPU API: **13.86x**. Maximum
absolute and relative errors were 1.526e-5 and 5.965e-6, within the declared
per-element gate. Its first call, which includes CUDA initialization and the
16 MiB Q8 upload, took 21.228 ms; that is not the steady-state metric.
GPU0 sampled at <=36 C and recovered under the required five-minute gate.
See [T05A evidence](../results/T05A-dn-qkv-2048x8192.md).

## What a T05A pass does and does not show

A pass shows that one cached Q8/FP32 projection benefits from the P40. It does
not prove an end-to-end Qwen speedup: one decoded token has 90 such calls,
and Qwen's recurrent/state work remains on the CPU. T05B will therefore add
only the three-projection DeltaNet call chain, preserve CPU fallback, measure
host-device transfers, and compare fixed Qwen output/logits before any full
generation claim. Tensor placement would initially alternate whole DeltaNet
layers between GPUs only if T05B shows a benefit; token-sequential layers make
two-device tensor parallelism an assumption to test, not a default.

## Explicit non-goals

- No generic `CUDA_DENSE`, `COLI_CUDA_ATTN`, Tensor Core, or MTP knob sweep.
- No `COLI_CUDA_TC_W4A16`: P40 / sm_61 has no Tensor Cores.
- No DP4A activation quantization, Q4 weights, LUT arithmetic, or model
  quality claim in this control.
- No modification to the production Colibri source or the separately dirty
  experimental source tree. A clean, dedicated worktree is required before
  source integration.
