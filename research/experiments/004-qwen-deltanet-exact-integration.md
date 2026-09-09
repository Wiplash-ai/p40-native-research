# E004 / T06 — Qwen exact-Q8 DeltaNet integration canary

Status: T06 and T06B are rejected because they poisoned the expert-tier CUDA
context. T06C passed its repaired-guard repeat with exact output and no runtime
diagnostic. It is eligible for one separately guarded 64-output comparison;
no sustained performance result is accepted yet.

## Hypothesis

The T05G exact-order Q8 kernel can accelerate Qwen's three DeltaNet Q8
projections without changing generated output when it caches the 90 matrices on
GPU0 and transfers only the FP32 activation/result for each single-row call.

## One change

`COLI_CUDA_DN_CPUORDER=1` is the only behavior change. The clean
`p40-t06-dn-cpuorder` worktree adds an opt-in `P40QwenDnTensor` cache to the
existing dense-Q8 registry. Only `dn_qkv`, `dn_z`, and `dn_out` are tagged.
The Q8 cache initializes after the existing two-GPU expert tier, uses GPU0,
reselects GPU0 before every CUDA call, and falls back permanently to the
existing CPU `matmul_q()` path after any failure.

`dn_b`, `dn_a`, convolution, recurrence, normalization, attention, MoE,
LM-head, expert placement, and all other dense projections are unchanged.

## Falsifying run

Run one fixed 16-output Qwen invocation under the existing dual-GPU 125 W
guard policy. The guard pins the experimental binary digest, model, prompt,
environment, two GPUs, output count, and expected 16-token stdout hash from
the accepted T04 control. It rejects any generated-output difference. It also
requires idle/cool preflight, BMC/fan checks, watchdog monitoring, release of
both GPUs, restored power limits, and the existing cooldown.

Promote only if:

1. the fixed output hash is identical;
2. the log confirms all 90 Q8 DeltaNet matrices became active on GPU0;
3. thermal and cleanup gates pass; and
4. the process exits normally.

Only after this canary passes may a separate 64-output timing comparison run.
It must keep the exact same configuration except for output length.

## T06 result and T06B correction

The first guarded attempt activated the 90-matrix cache and produced the exact
accepted stdout hash, but immediately emitted an invalid-resource-handle error
from the expert tier followed by 2,155 illegal-memory-access diagnostics. The
CPU fallback then preserved the text while MoE execution degraded to CPU
fallback (0.54 tok/s), so neither output identity nor the superficially lower
DeltaNet timer is evidence of a usable speedup. Thermal cleanup passed, but
the integration is rejected.

T06B changed one thing: the exact Q8 helper owned a nonblocking GPU0 stream
and used explicit stream synchronization. It still failed identically. Source
inspection then identified the cause: the helper's raw `cudaSetDevice(0)`
changed CUDA's current device without updating `backend_cuda.cu`'s thread-local
`g_current_device`. The backend therefore sometimes skipped a required switch
and launched a GPU1 stream while GPU0 was current.

T06C changes only device selection: the helper calls a new experimental
`coli_cuda_select_device()` API, which uses the backend's own context selector
and cache. Its first fixed 16-output run produced the accepted stdout hash,
reported 2.08 output tok/s, activated all 90 Q8 matrices, and emitted only the
two normal `[CUDA] device` inventory lines. It had no runtime CUDA failure,
custom-helper fallback, illegal access, or invalid resource-handle line.
The guard initially falsely classified those inventory lines as diagnostics.
Its line-aware correction permits only `[CUDA] device ...` inventory and
rejects every other `[CUDA] ...` line plus the helper fallback marker; local
guard tests cover both cases. The independently repeated canary passed: exact
stdout hash, no CUDA runtime diagnostic, 2.07 output tok/s over 15 decoded
steps, peak sampled GPU temperatures 46 C / 43 C, peak VRAM 8,855 / 7,895 MiB,
all fans 2,000–2,100 RPM, empty cards, and restored 250 W limits. Its 45.49
ms/token DeltaNet phase versus the T04 16-token control's 354.79 ms/token is
strong short-canary evidence, but only a separate 64-output comparison may
support an end-to-end timing conclusion.

## Static evidence

On 2026-09-08, with `CUDA_VISIBLE_DEVICES=""`, the clean remote worktree
compiled `qwen36` and the new `sm_61` helper. The Qwen dense-batch, context,
JSON-escaping, and cache-index CPU regression tests passed. This is build and
fallback evidence only; it is not model, CUDA-runtime, or performance proof.
