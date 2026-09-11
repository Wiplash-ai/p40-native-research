# Qwen3.6 P40 service deployment

`colibri-qwen36.service` intentionally runs a root-owned copy of the accepted
T24 source/binary path, not the mutable upstream checkout. The runtime binds
only `127.0.0.1:8000`; no API key is needed for that local boundary. A reverse
proxy, LAN binding, or public exposure requires an explicit authenticated
deployment decision.

The service uses the accepted exact-Q8 DeltaNet pair configuration. It does
not enable rejected W4A8, shared-pair, NUMA, recurrence-fusion, profiling, or
speculative paths. `--kv-slots 1` matches the direct Qwen engine's single-KV
execution model.

The copied launcher's `--gpu` check interrogates its unrelated GLM binary, so
this unit intentionally sets the verified Qwen CUDA variables in its
environment rather than passing `--gpu`. This avoids a false CPU-only refusal
without changing Colibrì source.

The 12.90 tok/s T24 result was collected under a 125 W/card benchmark guard.
This deployment must measure an actual OpenAI-compatible request before
claiming any rate, especially the requested 14 tok/s target.

## Live acceptance — 2026-09-11

The server unit is enabled and active. It loaded the exact Q8 cache and the
QKV/Z pair marker, completed the 10,240-expert VRAM warmstart, and exposes
`/health`, `/v1/models`, and `/v1/chat/completions` on `127.0.0.1:8000`.

A 64-completion-token streamed OpenAI request measured 1.630 s to first
content and 3.966 s from first content through `[DONE]`: 16.136 visible
completion tokens/s by that client framing. The same request was 11.436
completion tokens/s end to end, including prompt rendering and prefill. Peak
observed test temperatures were 50 C / 49 C; peak observed power was 104.27 W
/ 60.96 W, well below each card's 250 W limit. These are one-request service
acceptance measurements, not a capacity or concurrency claim.
