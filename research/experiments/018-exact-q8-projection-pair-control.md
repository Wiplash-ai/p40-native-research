# E018 / T23 - exact Q8 projection launch/synchronization control

Status: designed; intentionally not run during the post-T22 GPU break.

## Hypothesis

The largest remaining exact-Q8 stages lose meaningful time to the CPU-order
helper's per-projection boundary: host-to-device input copy, one kernel,
device-to-host output copy, then synchronization. In a DeltaNet block, the
2048-to-8192 QKV projection and the 2048-to-4096 Z projection share the same
hidden input and can be issued before either result is needed by the host.

## Smallest falsification

Build a GPU0-only, model-free control around the existing exact CPU-order
kernel and real DeltaNet pair shapes. Compare only:

1. Current serial boundary: upload hidden, QKV, download/synchronize; then
   upload the identical hidden vector, Z, download/synchronize.
2. Pair boundary: upload hidden once, launch the unchanged QKV kernel and
   unchanged Z kernel in one stream, download both outputs, synchronize once.

The subsequent CPU boundary and 4096-to-2048 output projection remain exactly
the same in both arms. A 30-layer cache-aware sweep must retain distinct Q8
weights and scales, so host cache locality cannot fabricate a win.

## Measurements and gates

- Bitwise equality of every pair output against the current GPU CPU-order
  kernel, plus the existing CPU reference check.
- CUDA event timings for upload, QKV kernel, Z kernel, download, and the
  host-observed completion time.
- Median and distribution across fixed repetitions; cached weights only.
- At least 15% lower pair completion latency, with no new allocation above the
  declared control budget, to justify a Qwen integration canary.

This is an exact launch/synchronization test, not a new arithmetic kernel.
If it fails, reject pair batching and assess CUDA Graph capture separately.
If it passes, integration must preserve the QKV/Z host boundary and use a new
exact-output 16-token Qwen canary before any 64-token comparison.
