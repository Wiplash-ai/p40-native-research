# T48 — production Colibrì Qwen3.6 API acceptance

Status: deployed and accepted. The service is local-only.

`colibri-qwen36.service` is enabled on the server and starts the immutable
release `/opt/colibri-qwen36-t24`, whose Qwen engine SHA-256 is
`009dd7daea34d65aba67aa442b79e7ff0d914642008ebd3b37b26a9f36a2ca64`.
It points to the pinned Kreuzzelg Qwen3.6-35B-A3B checkpoint and binds only
`127.0.0.1:8000`; no unauthenticated network listener was created.

The live log verified all deployment invariants:

- two Tesla P40s and the CUDA VRAM expert tier;
- all 10,240 experts warmstarted in VRAM;
- the exact Q8 cache (90 DeltaNet, LM head, 40 attention, 120 shared MLP
  matrices) on GPU 0;
- exact QKV/Z pair issue/join with CPU B/A overlap;
- OpenAI-compatible API ready at `/v1`.

`/health`, `/v1/models`, a non-streaming `/v1/chat/completions` request, and
an attached `coli chat` smoke test all passed. The API model id is
`qwen3.6-35b-a3b-colibri-i4-p40`.

One bounded 64-completion-token stream measured 1.630 s TTFT and 3.966 s from
first content to `[DONE]`, or 16.136 visible completion tokens/s by client
stream framing. End-to-end was 5.596 s / 11.436 completion tokens/s because it
includes prompt rendering and prefill. The requested 14 tok/s decode target is
met by the streamed-decode measurement; it is not met as an end-to-end prompt
latency claim.

The request peaked at 50 C / 49 C, 104.27 W / 60.96 W, and 9,725 MiB / 7,895
MiB on GPU 0 / GPU 1. The 70 C safety cutoff was not approached during that
request. A later idle-residency observation showed GPU 1 reaching 70 C at
about 56 W, which correctly stopped the service and released VRAM. The unit
is enabled but intentionally inactive pending cooldown; this blocks a
sustained-availability claim. Service configuration and reproducible commands
are in `deployment/`.
