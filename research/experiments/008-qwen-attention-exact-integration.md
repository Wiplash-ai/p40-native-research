# E008 / T12 — Qwen exact-Q8 attention-projection integration

Status: pass for the fixed 16-output guarded integration canary; a sustained
comparison has not yet run.

## One change

`colibri-t12-attention` is a new experimental source copy derived from T09,
never the production engine. Its opt-in `COLI_CUDA_ATTN_CPUORDER=1` flag marks
only the ten full-attention layers' Q/K/V/O Q8 matrices for the existing exact
CPU-order GPU helper. It validates exactly 40 such matrices at cache build.

It does not change the attention function's CPU-resident q/k normalization,
partial RoPE, KV writes, score dot products, softmax, value accumulation, or
output gate. It does not alter DeltaNet, LM-head, MoE routing, expert cache,
GPU count, or power policy.

## Static acceptance

The binary built with `CUDA_VISIBLE_DEVICES=""`, `CUDA_ARCH=sm_61`, and SHA-256
`cded60fd2b68980c858577354b1ada9fc89ea8bb813a95deb50b999fff6b7dad`.
The dense-batch, KV-context, JSON-escape, and cache-index tests passed with no
model or CUDA workload.

## Guarded canary result

The isolated binary completed a forced 16-output run (`run_id`
`8952b573-6b2d-454f-bac8-f307005f8f5a`) with the existing exact DeltaNet and
LM-head caches, the attention flag, both P40s capped at 125 W, and the fixed
public prompt. Its stdout SHA-256 exactly matched the established oracle:
`43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a`.

The required marker confirmed 90 DeltaNet matrices, the LM head, and all 40
attention Q/K/V/O matrices cached on GPU 0. Engine-reported speed was
4.98 tok/s (3.2 seconds for 16 output tokens; TTFT 0.92 seconds). Decode-stage
timing was 47.35 ms/token DeltaNet, 8.24 attention, 91.62 MoE, 3.53 LM head,
and 150.76 total. This turns attention from the 123.21 ms/token T09 bottleneck
into a small phase, but the 16-output canary is not a sustained comparison.

All experts remained resident, with zero CPU expert misses or swaps. Telemetry
sampled maximum temperatures of 42 C / 43 C and peak allocations of 8,155 /
7,893 MiB (GPU 0 / GPU 1); every fan stayed at 2,000--2,100 RPM. The guard
released allocations, cooled to its gate, and restored both 250 W power limits.

## Next experiment

After a physical no-workload break and fresh two-sample cool/idle preflight,
run a new hash-pinned 64-output T13 comparison that changes only `N_NEW`.
It must reproduce the established 64-output stdout hash, keep the exact
90+1+40 marker, and preserve the same thermal, fan, allocation-release, and
power-restore gates. Do not reinterpret the T12 short-canary rate as sustained
throughput.
