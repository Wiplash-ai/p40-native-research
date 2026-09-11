# S0/S1 — bounded swarm controller and first P40 executor baseline

Date: 2026-09-11
Status: Stage 0 pass; Stage 1 rate baseline pass, repeats and quality gate pending.

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
| Decode tokens / timing | 46 / 1.059334 s |
| Decode rate | **43.42 tok/s** |
| Prompt tokens / timing | 33 / 6.824077 s |
| Request wall time | 44.4663 s |
| GPU 0 peak | 5,713 MiB, 184.58 W, 46 C, 98% |
| GPU 1 peak | 3 MiB, 50.42 W, 38 C, 3% |
| Post-unload state | GPU 0: 0 MiB / 41 C; GPU 1: 0 MiB / 37 C |

This is a single cold request, not a median. The request-wall figure includes
model loading and must not be compared with the decode rate. GPU 0 was the
only card doing material inference work. The completion stopped at 46 decode
tokens despite a cap of 64, so future repetitions record actual decode count
and use a longer bounded workload for variance estimates.

## Decision

The first executor candidate clears the basic P40 fit/rate gate. It does not
yet clear tool-use, coding-task, diversity, concurrent-slot, or optimized
sidecar gates. Next:

1. repeat an exact stock benchmark after an idle/cool state to establish
   variance;
2. construct stock and Pascal-patched llama.cpp `sm_61` sidecars in an
   isolated directory using this exact GGUF blob;
3. compare them one setting at a time before changing the swarm executor
   adapter or Ollama service.

## Evidence

- `research/results/raw/S0-capacity-20260911T051119Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-stock-20260911T051931Z.json`
- `swarm/`, `tests/test_swarm_s0.py`, and `tests/test_swarm_ollama_bench.py`
