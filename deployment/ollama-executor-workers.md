# Candidate two-worker Ollama executor topology

Status: measured candidate only. No persistent unit has been installed.

Each worker must receive a single physical P40 and bind only loopback. On this
host, CUDA visibility alone is unsafe because Ollama can select Vulkan first.
Force the observed Pascal-compatible CUDA package as well.

| Worker | Bind | CUDA visibility | Required backend |
| --- | --- | --- | --- |
| executor-0 | `127.0.0.1:11441` | `0` | `cuda_v12` |
| executor-1 | `127.0.0.1:11442` | `1` | `cuda_v12` |

Shared settings:

```text
OLLAMA_MODELS=/mnt/ai-ssd/ollama/models
OLLAMA_NUM_PARALLEL=1
```

The Stage-2 warm control used `qwen3:8b`, 4,096 context, one request per
worker, and measured 87.46 aggregate decode tok/s. It also peaked at 52 C /
54 C. Before deployment, run the thermal preflight and establish service-level
stop/restart behavior; do not run these workers alongside the two-GPU Qwen
planner.

The existing system Ollama service remains separate and unchanged.
