# Local model runbook

This machine has three separate model lanes. They must stay separate: neither
model server binds to the LAN, and Qwen, K3, and temporary Ollama workers must
not be loaded together without a fresh thermal and VRAM preflight.

```text
OpenCode on this workstation
        |
        | authenticated private-LAN API (or SSH tunnel)
        v
192.168.1.194
  :8100  Wiplash local model router (starts/stops approved backends)
  :8000  Colibri Qwen3.6 35B-A3B, 2x P40 CUDA (existing service)
  :8001  Colibri Kimi K3, one-P40 Vulkan (template only; storage-gated)
  :11434 Ollama workers / executor models (existing service)
```

## Current state

| Lane | Endpoint on server | Status | Use now |
| --- | --- | --- | --- |
| Wiplash local model router | `192.168.1.194:8100/v1` | installed only after the deployment below | Yes; preferred OpenCode entry point |
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

### Preferred: one command

The normal workstation `opencode` CLI has the private-LAN router endpoint and
its local API-key file configured globally. Start it from any project:

```sh
cd /path/to/project
opencode
```

Use OpenCode's `/models` picker and choose from the `wiplash-router` entries:

- `wiplash-router/wiplash/qwen35b` checks both P40s are cool and empty,
  starts the accepted Qwen service on demand, and stops it after 120 seconds
  idle.
- `wiplash-router/wiplash/qwen3-8b` routes to the existing Ollama service and
  sends its native unload request after 180 seconds idle.
- `wiplash-router/wiplash/qwen3-coder-30b` does the same for the coder model.

The router no longer imposes a 512-token completion ceiling. The configured
windows are Qwen3.6 35B: 262,144 context / 16,384 output; Qwen3 8B: 40,960
context / 16,384 output; and Qwen3-Coder 30B: 16,384 context / 16,384 output.
Prompt and completion tokens share each model's context window, so a long
conversation naturally leaves fewer tokens available for one reply.

The router never runs Qwen and an Ollama model together. It refuses a Qwen
start above 50 C or if either GPU reports more than 256 MiB in use; Colibri's
own 70 C termination guard remains the final thermal containment layer.

The router binds only to the server's current private IPv4 address
`192.168.1.194`, accepts callers only from `192.168.1.0/24`, and requires its
bearer key on every endpoint. It is not a public API: use an SSH tunnel when
off the trusted LAN. It presents only its three fixed aliases and accepts only
`/v1/models`, `/v1/chat/completions`, and `/v1/completions`; it does not proxy
arbitrary URLs, models, or control commands.

### Direct backend mode

Open a tunnel in one terminal when you are off-LAN or need a direct backend for
diagnosis. The K3 forward is harmless while K3 is absent; it becomes useful
only after that service is installed.

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
- `ollama-p40/wiplash-qwen3-8b-40k:latest`
- `ollama-p40/qwen3-coder-normal-30b-16k:latest`

The model definitions live in
`~/.config/opencode/opencode.json`. The default remains the existing OpenRouter
model; select a local model explicitly with `/models` so a normal OpenCode
session does not load GPUs unexpectedly. The first selection must occur only
after its backend is healthy; OpenCode configuration itself does not start a
model.

### Qwen tool calling

The deployed Qwen3.6 gateway accepts OpenAI `tools` requests for the
`wiplash-router/wiplash/qwen35b` model. Its local Colibri branch
`wiplash/qwen36-tool-calling` at `2d6011d` renders the checkpoint's official
Qwen3.5/3.6 tool template, returns OpenAI-shaped tool calls, and renders tool
results back into the next model turn. A live guarded request produced
`finish_reason: "tool_calls"` for the declared `get_weather` function with
`{"city":"Rome"}`.

The i4 checkpoint can omit the outer `<tool_call>` tag while still producing a
complete `<function=DECLARED_NAME>...</function>` block. The gateway recovers
only that complete, explicitly declared form; it does not promote an arbitrary
function-looking string into a tool call. The router already forwards the
standard `tools` payload unchanged.

### Better Hashline edit transport

The global OpenCode configuration loads `opencode-better-hashline@0.9.0`. Its enforced
tool surface replaces the native edit/write/apply-patch tools with exact
snapshot-based `hashline_*` operations, which is useful for smaller open-weight
models that are more reliable when edits include stable line anchors. LSP
diagnostics are intentionally disabled, so enabling the plugin does not start
language servers from repository configuration.

The plugin has been loaded successfully with this workstation's OpenCode
`1.18.30`; a direct `hashline_read` returned a snapshot ID. Its optional
bundled verifier still pins itself to OpenCode `1.18.4`, despite the plugin's
declared `>=1.18.3 <2` range. Do not downgrade the working client just to
satisfy that stale verifier. Treat Hashline as better edit transport—not proof
that a model's code quality has improved—and keep normal tests/reviews in the
loop.

### Clipboard on this Wayland workstation

OpenCode's selection and message-copy actions require `wl-copy` / `wl-paste`
on Wayland. A user-local `wl-clipboard` copy lives under
`~/.local/opt/wl-clipboard`; the normal `opencode` wrapper places it on `PATH`
before launching the standard CLI. No separate OpenCode launcher is required.
Select a response or use the configured message-copy shortcut
(`Ctrl+Shift+C` or leader then `y`); the clipboard bridge receives the copied
text. A system package installation of `wl-clipboard` is an equivalent future
replacement.

### Permissions and Codex skill mirror

The global OpenCode configuration uses `"permission": "allow"`, so normal
sessions receive read, glob, grep, list, bash, subagent, external-directory,
web fetch, web search, LSP, questions, and skills without approval prompts.
Better Hashline deliberately replaces the native edit/write/patch tools with
its snapshot-bound `hashline_*` mutations. The regular `opencode` launcher
also opts into its Exa web-search integration.

The compatible user-owned Codex skills are copied to
`~/.config/opencode/skills` and advertised through OpenCode's native `skill`
tool. This mirror intentionally excludes the Google/Gemini and Twilio skills;
it neither installs nor configures any MCP server. Refresh it after changing a
source skill:

```sh
/home/jordanculver/Laboratory/p40-native-research/scripts/sync-opencode-skills
```

## Install or update the local model router

The following is the only deployment path. It installs the router code as the
unprivileged `jordanculver` service, a root-owned fixed Qwen control helper,
and a sudo rule that permits exactly `qwen35b start`, `stop`, and `is-active`.
It also creates a private bearer key shared only with this workstation. It does
not start Qwen, Ollama models, or K3.

```sh
cd /home/jordanculver/Laboratory/p40-native-research
install -d -m 0700 ~/.config/wiplash
test -f ~/.config/wiplash/model-router-api-key || \
  (umask 077 && openssl rand -hex 32 > ~/.config/wiplash/model-router-api-key)
ssh jordanculver@192.168.1.194 'mkdir -p /tmp/wiplash-model-router-stage'
scp -r model_router deployment/wiplash-model-router.json \
  deployment/wiplash-model-router.service deployment/wiplash-model-control \
  deployment/wiplash-model-router.sudoers deployment/ollama-qwen3-8b-40k.Modelfile \
  ~/.config/wiplash/model-router-api-key \
  jordanculver@192.168.1.194:/tmp/wiplash-model-router-stage/
ssh jordanculver@192.168.1.194 '
  sudo install -d -o root -g root -m 0755 /opt/wiplash-model-router &&
  OLLAMA_HOST=172.17.0.1:11434 /usr/local/bin/ollama create wiplash-qwen3-8b-40k:latest -f /tmp/wiplash-model-router-stage/ollama-qwen3-8b-40k.Modelfile &&
  sudo install -o root -g root -m 0644 /tmp/wiplash-model-router-stage/wiplash-model-router.json /etc/wiplash-model-router.json &&
  sudo sh -c '\''umask 077; IFS= read -r key < /tmp/wiplash-model-router-stage/model-router-api-key; printf "WIPLASH_MODEL_ROUTER_API_KEY=%s\\n" "$key" > /etc/wiplash-model-router.env'\'' &&
  sudo install -o root -g root -m 0755 /tmp/wiplash-model-router-stage/wiplash-model-control /usr/local/sbin/wiplash-model-control &&
  sudo install -o root -g root -m 0440 /tmp/wiplash-model-router-stage/wiplash-model-router.sudoers /etc/sudoers.d/wiplash-model-router &&
  sudo visudo -cf /etc/sudoers.d/wiplash-model-router &&
  sudo cp -a /tmp/wiplash-model-router-stage/model_router /opt/wiplash-model-router/ &&
  sudo install -o root -g root -m 0644 /tmp/wiplash-model-router-stage/wiplash-model-router.service /etc/systemd/system/wiplash-model-router.service &&
  sudo systemctl daemon-reload &&
  sudo systemctl enable --now wiplash-model-router.service
'
curl --fail -H "Authorization: Bearer $(<~/.config/wiplash/model-router-api-key)" \
  http://192.168.1.194:8100/healthz
curl --fail -H "Authorization: Bearer $(<~/.config/wiplash/model-router-api-key)" \
  http://192.168.1.194:8100/v1/models
```

After changing the source, rerun that procedure. Before the next deployment,
run the offline test suite below. The service has no model-load side effect:
the authenticated `GET /healthz` and `GET /v1/models` requests above are the
post-install smoke tests.

```sh
python3 -m unittest tests/test_model_router.py -v
```

Ollama documents its native `keep_alive: 0` API control as an immediate unload.
The router uses that documented endpoint only after its own idle timer, so it
does not ask the OpenAI-compatible client to understand an Ollama-only field.

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
