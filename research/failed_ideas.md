# Rejected, deferred and untested ideas

No architecture has been trained or falsified yet. Do not label proposed candidates failed before tests.

| Idea | Current decision | Evidence/revisit condition |
|---|---|---|
| Raise CUDA_EXPERT_GB to 24/32 to exploit 48 GB | Reject for current model | Whole expert payload 15.117 GiB; setting is per GPU. Revisit for measured larger tensor residency after implementation |
| CUDA_DENSE/ATTN/PIPE flags already move Qwen trunk | Rejected by source | Implement explicit dispatch and capability reporting |
| COLI_NUMA=1 proves effective Qwen placement | Rejected as evidence | Inspect numa_maps and actual OS policy |
| More HTTP clients equals continuous batching | Rejected for installed Qwen | One KV slot and serial C serve loop |
| CUDA memory residency means GPU-only inference | Rejected by source | CPU owns trunk/state/head and MoE coordination |
| Switch to CUDA13/compiler or newest PyTorch/Triton for performance | Defer | Preserve sm61 working tools; modern builds may omit Pascal |
| Force W4A16 Tensor Cores on P40 | Reject | Hardware has no Tensor Cores; retain only ordinary FP16-storage experiments |
| Replace INT4 with ternary without training/calibration | Reject as a quality-preserving change | Train small candidate or evaluate calibration/quality explicitly |
| Sparsity or lookup tables always beat GEMV | Untested | Include indexing/lookup/reduction/build overhead and cold data |
| 4 P40 guarantees a faster 35B model | Untested, weak capacity argument | Experts already fit one; profile compute and communication first |
| Kimi K3 10–20 tok/s follows from Qwen speedup | Reject as an extrapolation | Capacity, active bytes, engine layout and actual tiers need separate analysis |
| Large-scale model training now | Defer | First prove correctness, learning and stable cooling on tiny cases |
