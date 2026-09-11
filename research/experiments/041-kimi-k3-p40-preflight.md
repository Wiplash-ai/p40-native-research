# S11 — Kimi K3 P40 Vulkan preflight

Date: 2026-09-11  
Status: backend and tiny-oracle pass; full-model measurement blocked by storage

## Hypothesis

The current NVIDIA driver, two Tesla P40 GPUs, and Colibri source can build
and execute the Kimi K3 Vulkan implementation correctly enough to justify a
future full-checkpoint trial.

## Control

- Colibri source: server commit `12a5c46`, copied to an isolated worktree at
  `/mnt/ai-ssd/p40-kimi-k3-preflight`.
- Qwen service inactive; no Ollama model resident.
- Added Vulkan build/runtime tools (`glslc`, `libshaderc1`, `libvulkan-dev`,
  and `vulkan-tools`) and a NumPy virtual environment only inside the isolated
  preflight worktree.
- Verified Vulkan enumerates both Tesla P40 cards under NVIDIA driver
  `580.173.02`.

## Result

`make VK=1 kimi-k3-tiny-check` passed the deterministic tiny K3 fixture's
greedy, chunked, teacher-forced, corruption-oracle, and recurrent checkpoint
tests. A direct raw-token invocation initialized the Vulkan backend on Tesla
P40 GPU 0 and generated eight tiny-fixture tokens. The fixture reported
621.04 decode tokens/s, but its 1.7 MB, six-layer model is a backend smoke
test: that number has no predictive value for 2.8T K3.

The source establishes an important topology constraint. K3 is a Vulkan
engine, not Qwen's CUDA tier. Its generic backend contains a second-device
expert primitive, but `kimi_k3.c` does not call it, so the observed engine
path is one physical P40. There is no 48 GB aggregate-VRAM claim.

## Blocker

No Kimi K3 checkpoint exists on the server. The documented original source
checkpoint is about 1.56 TB; `/mnt/ai-ssd` had about 178 GiB free. A full K3
tokens/s run and an enabled K3 API service are therefore not possible now.

## Decision

Accept the backend as buildable and correct on a tiny oracle. Do not infer
full-model quality or performance. Require a dedicated, fast, at-least-2-TB
local volume and a complete checkpoint before doing a 32-token thermal-
monitored full-model probe. The separate, loopback-only systemd template is
prepared in `deployment/` but deliberately not installed or enabled.

## Evidence

- `research/results/raw/S11-kimi-k3-p40-preflight-20260911T143500Z.json`
- `/mnt/ai-ssd/p40-kimi-k3-preflight` on the server (isolated build/test
  worktree; not production Colibri source)
