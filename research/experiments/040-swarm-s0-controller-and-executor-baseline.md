# S0/S1 — bounded swarm controller and first P40 executor baseline

Date: 2026-09-11
Status: Stage 0 pass; Stage 1 rate/plan gates pass; Stage 4 bounded fixture
pass. Meaningful repository-task reliability and multi-slot quality gates
remain pending.

## Hypotheses

1. A controller can preserve task, branch, and evidence boundaries without
   possessing autonomous execution authority.
2. A small Q4 8B executor fits and decodes quickly on one P40 at a 4K context,
   leaving the other P40 untouched.
3. A controller-rendered exact text replacement is a more reliable small-model
   action surface than a model-authored unified diff.

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

## S2 two-worker P40 topology

The physical-slot hypothesis is now measured, rather than assumed. Two stock
llama.cpp sidecars explicitly pinned to separate P40s produced 45.32 and 45.14
tok/s at the same time; their 128 tokens completed at 83.66 aggregate wall
tok/s. Both models used 5,661 MiB and peaked at 46 C / 47 C. This establishes
that PCIe does not materially slow two independent small-model decode streams.

The equivalent Ollama topology required an additional backend constraint. A
temporary server with only `CUDA_VISIBLE_DEVICES=1` selected Vulkan first and
ran on physical GPU 0 at 26.27 tok/s. Adding `OLLAMA_LLM_LIBRARY=cuda_v12`
made the same isolated server enumerate only physical GPU 1 through CUDA and
produce 43.18 tok/s. `CUDA_VISIBLE_DEVICES` is therefore insufficient by
itself on this host.

Two temporary, loopback-only Ollama servers with these settings were then
tested at once:

```text
worker 0: CUDA_VISIBLE_DEVICES=0, OLLAMA_LLM_LIBRARY=cuda_v12, 127.0.0.1:11441
worker 1: CUDA_VISIBLE_DEVICES=1, OLLAMA_LLM_LIBRARY=cuda_v12, 127.0.0.1:11442
OLLAMA_MODELS=/mnt/ai-ssd/ollama/models
OLLAMA_NUM_PARALLEL=1
```

Cold model-load wall time was 18.33 s, so its 5.02 wall tok/s figure is not a
throughput result. Both runners nevertheless decoded at 43.41 / 43.54 tok/s
on their intended GPUs with identical outputs. After an explicit one-token
preload, each worker remained resident at 5,713 MiB. A warm concurrent pass
decoded 38 tokens at 43.73 tok/s and 38 at 44.09 tok/s: 87.46 aggregate decode
tok/s (`76 / max(0.86894, 0.86187)`). Its end-to-end concurrent request rate
was 48.36 tok/s because prompt processing and API work remain on the request
critical path. GPU peaks were 52 C / 54 C, below the 70 C abort ceiling; all
VRAM was released on teardown.

This is the accepted executor-host topology for later quality tests, not a
deployed service. Qwen stays stopped while two executor workers occupy the
cards; no concurrent Qwen-plus-worker fit claim has been made.

## Schema-only live executor plan

`qwen3:8b` returned valid JSON for the fixed five-field planning schema under
Ollama's `format` constraint. It selected the allowlisted `python-unittest`
profile and returned no command. The plan request used 120 prompt and 87 decode
tokens, peaked at 5,713 MiB / 185.45 W / 47 C on GPU 0, then unloaded to 0 MiB.
This validates structured planning only: it did not receive a repository,
worktree, shell, SSH credential, or dispatch capability.

## S3/S4 bounded patch-quality result

The first real executor-quality fixture was an isolated Git repository with an
inverted `is_even` predicate and four unit assertions. Its baseline
`python-unittest` profile failed. The model received only the two supplied
source files. It could not choose a shell command, access the repository, or
write outside a controller-created detached worktree. GPU start was 41 C;
the 70 C sampled abort threshold was never approached. Colibri remained
inactive throughout.

S3 asked `qwen3:8b` for a schema-constrained unified patch. It selected the
correct semantic replacement, but emitted `@@ -1,4 +1,4 @@` for a two-line
file. `git apply --check` correctly rejected the patch as corrupt. The model
used 247 prompt and 128 decode tokens at 40.07 decode tok/s, and GPU 0 peaked
at 5,713 MiB, 186.53 W, and 49 C. This is a failed action-protocol result, not
a model-correctness pass. The patch route stays experimental.

S4 replaced the free-form diff with one schema-constrained exact text
replacement. Host validation requires a relative supplied-file path, distinct
non-empty strings, and exactly one occurrence of `expected_text` before any
worktree write. The controller renders that replacement and independently
runs `git-diff-check` and `python-unittest`; the model's requested validation
profile is not authority to skip either check.

`qwen3:8b` proposed the correct single replacement in `calculator.py`. The
baseline failed, `git-diff-check` passed in 4 ms, and all unit assertions
passed in 69 ms. This request used 256 prompt and 68 decode tokens at 38.53
decode tok/s; GPU 0 peaked at 5,713 MiB, 182.94 W, and 46 C, then unloaded to
0 MiB. This clears one trivial fixture only. It does not establish real-repo
success rate, safe scope selection, model diversity, or multi-worker quality.

## Decision

The first executor candidate clears the P40 fit/rate, schema-plan, and one
controller-rendered text-edit fixture gate. It does not yet clear repository
task reliability, tool-use, diversity, concurrent-slot quality, or
optimized-sidecar gates. Next:

1. run a fixed multi-task repository corpus through controller-rendered,
   worktree-isolated text edits, with the controller reviewing evidence before
   promotion;
2. test corpus throughput and quality against the accepted two-worker topology
   only after single-worker reliability is measured;
3. keep the Pascal-MMQ fork out of the executor adapter unless a future,
   controlled measurement clears a meaningful reproducible threshold.

## Evidence

- `research/results/raw/S0-capacity-20260911T051119Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-stock-20260911T051931Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-stock-repeat1-20260911T052457Z.json`
- `research/results/raw/S1-llama-base-server-20260911T054131Z.json`
- `research/results/raw/S1-llama-pascal-server-20260911T054231Z.json`
- `research/results/raw/S1-ollama-qwen3-8b-schema-plan-20260911T054614Z.json`
- `research/results/raw/S2-dual-sidecar-qwen3-8b-20260911T055045Z.json`
- `research/results/raw/S2-ollama-qwen3-8b-cuda-gpu1-20260911T055447Z.json`
- `research/results/raw/S2-dual-ollama-qwen3-8b-20260911T055646Z.json`
- `research/results/raw/S2-dual-ollama-qwen3-8b-warm-20260911T055927Z.json`
- `research/results/raw/S3-ollama-qwen3-8b-unified-patch-20260911T133112Z.json`
- `research/results/raw/S4-ollama-qwen3-8b-text-edit-20260911T133204Z.json`
- `swarm/`, `tests/test_swarm_s0.py`, and `tests/test_swarm_ollama_bench.py`
