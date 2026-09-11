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

The production Qwen unit reserves the checkpoint's native 262,144-token
context and permits up to 32,768 completion tokens. Colibri clamps a client's requested output length
to `--ngen`; it does not generate that many tokens unless the model reaches
the limit rather than EOS. Long prompts and completions share the same 262K
window.

The copied launcher's `--gpu` check interrogates its unrelated GLM binary, so
this unit intentionally sets the verified Qwen CUDA variables in its
environment rather than passing `--gpu`. This avoids a false CPU-only refusal
without changing Colibrì source.

The 12.90 tok/s T24 result was collected under a 125 W/card benchmark guard.
This deployment must measure an actual OpenAI-compatible request before
claiming any rate, especially the requested 14 tok/s target.

## Live acceptance — 2026-09-11

The enabled server unit loaded the exact Q8 cache and the QKV/Z pair marker,
completed the 10,240-expert VRAM warmstart, and exposed `/health`,
`/v1/models`, and `/v1/chat/completions` on `127.0.0.1:8000`.

A 64-completion-token streamed OpenAI request measured 1.630 s to first
content and 3.966 s from first content through `[DONE]`: 16.136 visible
completion tokens/s by that client framing. The same request was 11.436
completion tokens/s end to end, including prompt rendering and prefill. Peak
observed test temperatures were 50 C / 49 C; peak observed power was 104.27 W
/ 60.96 W, well below each card's 250 W limit. These are one-request service
acceptance measurements, not a capacity or concurrency claim.

The systemd unit now launches through `colibri-thermal-supervisor.py`. It
refuses to load the model at 70 C or above. If either card reaches 70 C while
resident, it sends the launcher SIGTERM, gives it 45 seconds to release the
engine, kills only if needed, and exits successfully so systemd does not
restart into a hot chassis. Child crashes still propagate failure and retain
the ordinary restart policy.

At the next idle-residency observation GPU 1 rose to 70 C on about 56 W,
and the guard deliberately stopped the service. The unit is therefore enabled
but intentionally inactive until the cards cool and an operator starts it.
This is a safety result, not an availability claim.

```sh
sudo systemctl start colibri-qwen36.service
sudo systemctl status colibri-qwen36.service --no-pager
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

## Chassis cooling guard

The passive P40s rely on chassis airflow. This BMC's zone command uses a
0–100 domain: it read back `0x64` after a direct `0xff` write, so its existing
100% configuration is correctly sent as `0x64`. The fan guard is not the cause
of the present 2,000–2,100 RPM readings; sustained production use remains
blocked on physical airflow diagnosis or a safe thermal policy change.

## Local model router

`wiplash-model-router.service` is a separate, unprivileged, private-LAN
OpenAI-compatible router. It binds only to `192.168.1.194`, restricts callers
to `192.168.1.0/24`, and requires a bearer key on every endpoint. The router
maps three fixed aliases to the accepted Qwen service or known Ollama models,
prevents Qwen/Ollama co-residency, and releases a model after its short idle
timer. Its root-owned helper can execute only Qwen start, stop, and status
operations; it cannot accept arbitrary systemd units or commands.

`NoNewPrivileges` is intentionally not set on this unit: it would prevent the
router from invoking that fixed, sudoers-limited helper. The router process
still runs as `jordanculver`; the helper is the only intended privilege
transition.

See [the local model runbook](../docs/LOCAL_MODEL_RUNBOOK.md#install-or-update-the-local-model-router)
for the reviewed installation procedure and its non-model-loading smoke test.
