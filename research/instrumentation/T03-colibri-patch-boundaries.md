# T03 proposed experimental instrumentation boundaries

This is a patch plan only. No Colibri checkout was opened or modified for T03.

## Isolated patch surface

- `c/qwen36_tier.{c,h}`: add a per-request `qwen36_tier_metrics` owned by the tier; zero it at request start and serialize only after direct execution completes. Count actual successful CUDA launches, resident expert hits, CPU-cache misses, failed issues, and per-device H2D/D2H bytes. A failed issue must increment `failed_issue`, never `launches`.
- `c/backend_cuda.cu`: bracket only `coli_cuda_expert_group_issue` and `coli_cuda_expert_group_take` with optional CUDA events on the existing single stream/device. Record event elapsed time in nanoseconds after take/synchronization. Event creation, record, synchronization, and destruction failures fall back to a marked `timing_status: unsupported`; they must not alter expert results or fallback behavior.
- `c/qwen36.c`: retain `COLI_TIMERS`, adding non-additive interval records around allocation, DeltaNet projection/recurrence/output, attention, routing, shared expert, head, issue/copy/sync. Each record declares `relation: none|subset|overlap`; reports must not sum subset or overlap rows.
- Direct `qwen36` report writer only: emit the JSONL prefixes defined by `scripts/colibri_t03.py`. Do not claim served gateway output has this coverage.

## Invariants and fixture plan

- Instrumentation defaults off; no counters/events/allocation beyond a null metrics pointer, and token output must be byte-for-byte unchanged in existing CPU fixtures.
- Preserve one stream per device. Pair `issue`/`take` event ownership with the existing outstanding request slot; test issue failure, CPU fallback, repeated take, and scratch lifetime before any benchmark.
- Counter units are `count` and `bytes`; elapsed durations are signed-safe non-negative integer `ns`. Per-device rows include device ordinal plus UUID when the runner supplies it.
- T00 must pass before a guarded tiny GPU fixture tests CUDA events. That work is outside this T03 source-only acceptance.
