# Local model runbook

This machine has three separate model lanes. They must stay separate: neither
model server binds to the LAN, and Qwen, K3, and temporary Ollama workers must
not be loaded together without a fresh thermal and VRAM preflight.

```text
OpenCode on this workstation
        |
        | SSH tunnel only
        v
192.168.1.194
  :8000  Colibri Qwen3.6 35B-A3B, 2x P40 CUDA (existing service)
  :8001  Colibri Kimi K3, one-P40 Vulkan (template only; storage-gated)
  :11434 Ollama workers / executor models (existing service)
```

## Current state

| Lane | Endpoint on server | Status | Use now |
| --- | --- | --- | --- |
| Colibri Qwen3.6 35B-A3B | `127.0.0.1:8000/v1` | installed, enabled, stopped | Yes, after a cool-start check |
| Ollama | `172.17.0.1:11434/v1` | installed | Yes, with one model/GPU at a time until cooling is upgraded |
| Colibri Kimi K3 | `127.0.0.1:8001/v1` | not installed or enabled | No: full checkpoint storage is absent |

The existing Qwen service is deliberately not started by this runbook. It uses
the accepted Q24 configuration and an all-GPU 70 C thermal cutoff. Check it
before every use:

```sh
ssh jordanculver@192.168.1.194 \
  'nvidia-smi --query-gpu=index,temperature.gpu,memory.used,power.draw --format=csv,noheader'
ssh jordanculver@192.168.1.194 'sudo systemctl start colibri-qwen36.service'
ssh jordanculver@192.168.1.194 'systemctl status colibri-qwen36.service --no-pager'
```

Stop it when finished, especially before running a P40-pinned Ollama worker:

```sh
ssh jordanculver@192.168.1.194 'sudo systemctl stop colibri-qwen36.service'
```

## OpenCode

OpenCode `1.18.30` is installed on this workstation. The wrapper below adds
only local-model providers; it does not overwrite the existing global
OpenRouter configuration.

Open a tunnel in one terminal. The K3 forward is harmless while K3 is absent;
it becomes useful only after that service is installed.

```sh
ssh -N \
  -L 8000:127.0.0.1:8000 \
  -L 8001:127.0.0.1:8001 \
  -L 11434:172.17.0.1:11434 \
  jordanculver@192.168.1.194
```

In a second terminal, from this repository:

```sh
./scripts/opencode-local-models models colibri-p40
./scripts/opencode-local-models models ollama-p40
cd /path/to/project
/home/jordanculver/Laboratory/p40-native-research/scripts/opencode-local-models
```

Then use OpenCode's `/models` picker and select one of:

- `colibri-p40/qwen3.6-35b-a3b-colibri-i4-p40`
- `ollama-p40/qwen3:8b`
- `ollama-p40/qwen3-coder-normal-30b-16k:latest`

The model definitions live in
`opencode/wiplash-local-models.json`. They are a separate configuration merged
through `OPENCODE_CONFIG`, so the global settings and credentials are not
rewritten. The first selection must occur only after its backend is healthy;
OpenCode configuration itself does not start a model.

## Kimi K3: completed preflight and current block

The actual Colibri K3 source was checked in an isolated server worktree.
Its tiny checkpoint passed deterministic greedy, chunked, teacher-forced,
oracle, and state-checkpoint tests. A Vulkan build also initialized **Tesla
P40 GPU 0** and decoded eight tiny-model tokens. This verifies the engine,
the NVIDIA Vulkan driver, and the one-device Vulkan path—not full K3 speed.

The full `moonshotai/Kimi-K3` source checkpoint is about **1.56 TB** (the
routed experts alone are about 1.45 TB). This server has only about 178 GiB
free on `/mnt/ai-ssd` and no K3 checkpoint. Therefore no meaningful full-model
tokens/sec number exists yet and a real service would be deceptive.

K3 uses a different execution path from Qwen:

- It is a Colibri **Vulkan** engine (`make VK=1 kimi_k3`), not the Qwen CUDA
  tier.
- The tested K3 engine chose P40 GPU 0. Although Colibri's generic Vulkan
  backend has a second-device primitive, current `kimi_k3.c` does not wire it;
  do not count on both P40s or 48 GB aggregate VRAM.
- K3's top-16 routed experts stream from the original MXFP4 checkpoint. Fast
  storage and expert I/O are expected to matter at least as much as GPU speed.
- `kv-slots=1` is intentional: the current K3 family declaration permits one
  KV slot. This is not an agent-swarm serving configuration.

### Storage and deployment gate

Before any K3 download or service installation, provide a dedicated fast
local NVMe volume with at least 2 TB usable capacity (more is preferable for
conversion/repacking headroom). Then, in this order:

```sh
# 1. Build in an isolated copy or the dedicated /opt release tree.
make -C /opt/colibri-kimi-k3/c VK=1 kimi_k3

# 2. Check the exact source model is complete and readable.
test -f "$K3_MODEL_DIR/config.json"
find "$K3_MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l

# 3. Keep the first API test small and locally bound.
COLI_MODEL="$K3_MODEL_DIR" /opt/colibri-kimi-k3/c/coli doctor --model "$K3_MODEL_DIR"

# 4. Run a 32-token thermal-monitored decode before installing/enabling a unit.
nvidia-smi --query-gpu=index,temperature.gpu,memory.used,power.draw --format=csv,noheader
```

Only after that measurement should the templates in `deployment/` be copied
to `/etc/colibri/kimi-k3.env` and
`/etc/systemd/system/colibri-kimi-k3.service`, followed by `daemon-reload` and
a manual start. Keep it disabled until the initial model-load and decode
measurement succeed. Never run it beside Qwen or an active Ollama GPU worker.

## Thermal rule

The service supervisor refuses a start at 70 C or higher and terminates its
child if either P40 reaches 70 C. That is a containment threshold, not a
target operating temperature. With the current cooling, start from a cool,
empty GPU state and stop the workload if temperatures rise persistently.
