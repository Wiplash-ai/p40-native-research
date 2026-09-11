# S0/S1 — bounded swarm controller and first P40 executor baseline

Date: 2026-09-11
Status: Stage 0 pass; Stage 1 rate/plan gates pass; repository-task and
multi-slot gates pending.

## Hypotheses

1. A controller can preserve task, branch, and evidence boundaries without
   possessing autonomous execution authority.
2. A small Q4 8B executor fits and decodes quickly on one P40 at a 4K context,
   leaving the other P40 untouched.

## Fixed conditions

- Controller binds only loopback and dispatch returns `409`.
- The harness accepts only a fixed static-check or Python-unittest profile in
  a detached Git worktree.
- Model host: Ollama `0.30.9`, `qwen3:8b`, 4,096 token context, `think=false`,
  seed 42, temperature 0, completion cap 64, `keep_alive=0s`.
- Start rejection: either P40 at or above 70 C. A run fails if its sampled
  peak reaches 70 C or GPU residency is not observed.
- Qwen/Colibri production service remained inactive for the measurement.

## Stage 0 controller result

`python3 -m unittest tests.test_swarm_s0 tests.test_swarm_ollama_bench -v`
passed all 7 tests. A real loopback server health check also returned the
expected dispatch-disabled Stage-0 response. The capacity probe found both
P40s empty at 35 C and 37 C and `/mnt/ai-ssd` with 185 GiB free before the
executor pull.

The implemented surface is intentionally insufficient for an agent to make a
production change: task creation, branch planning records, append-only
evidence, worktree-scoped fixed profiles, and promotion gates exist; model
dispatch does not.

## S1 stock Ollama result: `qwen3:8b`

| Metric | Result |
| --- | ---: |
| Cold decode, run 1 | 46 / 1.059334 s = **43.42 tok/s** |
| Cold decode, repeat | 46 / 1.051219 s = **43.76 tok/s** |
| Decode variation | 0.8% across the two observations |
| Prompt tokens / timing, repeat | 33 / 0.157328 s |
| Request wall time | 44.4663 s first run; 13.72 s repeat |
| GPU 0 peak | 5,713 MiB, 185.17 W, 46 C, 98% |
| GPU 1 peak | 3 MiB, 50.51 W, 38 C, 5% |
| Post-unload state | GPU 0: 0 MiB / 42 C; GPU 1: 0 MiB / 41 C |

This is two cold requests, not a sufficient distribution for a production
SLO. The request-wall figures include model loading and must not be compared
with decode rate. GPU 0 was the only card doing material inference work. The
completion stopped at 46 decode tokens despite a cap of 64, so later sidecar
tests use a server that reports exactly 64 predicted tokens.

## Isolated Pascal-MMQ sidecar A/B

The `pascallama.cpp` patch did not apply to current llama.cpp. Its preimage
matched upstream `e920c523e3b8a0163fe498af5bf90df35ff51d25`, immediately
before upstream's MMQ configuration refactor. Two separate `sm_61` builds at
that commit used the same flags (`GGML_CUDA`, forced MMQ, all quantisations,
and CUDA architecture 61), exact 5.2 GiB Qwen3 GGUF, GPU 0, 4K context, and a
64-token cap. The only source difference was the 15-line Pascal patch.

| Variant | 64-token decode | Exact output | GPU 0 peak | Decision |
| --- | ---: | --- | --- | --- |
| Stock MMQ | 45.40 tok/s | baseline | 5,513 MiB, 44 C | control |
| Pascal `mmq_y=96` + unroll | 45.52 tok/s | SHA-256 identical | 5,513 MiB, 45 C | reject |

The 0.27% rate difference is below the two-run noise floor and is not an
optimization. The sidecar's raw completion route is not chat-template
equivalent to Ollama, so its 45.40 tok/s cannot establish a product-level
Ollama replacement either.

## Schema-only live executor plan

`qwen3:8b` returned valid JSON for the fixed five-field planning schema under
Ollama's `format` constraint. It selected the allowlisted `python-unittest`
profile and returned no command. The plan request used 120 prompt and 87 decode
tokens, peaked at 5,713 MiB / 185.45 W / 47 C on GPU 0, then unloaded to 0 MiB.
This validates structured planning only: it did not receive a repository,
worktree, shell, SSH credential, or dispatch capability.

## Decision

The first executor candidate clears the P40 fit/rate and schema-plan gates. It
does not yet clear repository-task, tool-use, diversity, concurrent-slot, or
optimized-sidecar gates. Next:

1. run the fixed small repository corpus through worktree-isolated executor
   attempts, with the controller still reviewing evidence before promotion;
2. test a long-context prompt path and one concurrency configuration before
   treating a sidecar or a second GPU as an executor scaling path;
3. keep the Pascal-MMQ fork out of the executor adapter unless a future,
   controlled measurement clears a meaningful reproducible threshold.

## Evidence

- `research/results/raw/S0-capacity-20260911T051119Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-stock-20260911T051931Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-stock-repeat1-20260911T052457Z.json`
- `research/results/raw/S1-llama-base-server-20260911T054131Z.json`
- `research/results/raw/S1-llama-pascal-server-20260911T054231Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-schema-plan-20260911T054614Z.json`
- `swarm/`, `tests/test_swarm_s0.py`, and `tests/test_swarm_ollama_bench.py`
