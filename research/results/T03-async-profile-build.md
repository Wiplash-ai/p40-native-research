# T03 — Qwen asynchronous expert profiling build

Date: 2026-09-08. Status: **build pass; runtime measurement pending**.

The installed Qwen decode path uses `coli_cuda_expert_group_issue()` and
`coli_cuda_expert_group_take()`. The prior `COLI_CUDA_PROFILE` instrumentation
only covered the synchronous group wrapper, so it could not report the work
that Qwen actually schedules asynchronously.

The experimental-only patch
[`patches/0001-qwen36-async-expert-profile.patch`](../../patches/0001-qwen36-async-expert-profile.patch)
adds four persistent CUDA events to each context's one-outstanding-group slot:

1. before descriptor/input H2D work;
2. after H2D work;
3. after expert kernels; and
4. after D2H work.

After the existing `take()` stream synchronization, the patch adds the three
elapsed intervals to the already-reported `qt_stats()` group counters. It is
strictly opt-in through `COLI_CUDA_PROFILE=1`; event creation/timing failures
disable profiling for that group without changing inference or fallback.

## Validation completed

- The patch validates with `git apply --check` against the pinned server source
  `12a5c464b5c1f8292d578c62458706bc32d6ac95`.
- It was applied only to `/home/jordanculver/p40-native-research/colibri-engine`.
  Production `/home/jordanculver/colibri_engine` remained clean.
- Experimental engine build passed:
  `make -C c qwen36 CUDA=1 CUDA_ARCH=sm_61 NVCC=/usr/bin/nvcc`.
- Experimental binary SHA-256:
  `d7e8a284a4f864faddd4df6c16c1fbd3c8176222375fb23725af53c453f69051`.
- Running that binary without a model exited normally with the expected
  launcher guidance. No CUDA context, model weights, or inference run occurred.

## Pending acceptance

The patch has not yet been run with the model, so no event timing is claimed.
The next task is a new, allowlisted guarded Qwen model canary with a short fixed
generation and two-GPU power/thermal telemetry. Only that run can establish
whether CPU overlap, H2D/D2H, or the expert kernels dominate.
