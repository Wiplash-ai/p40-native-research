# Hardware evidence and workload gates

## Read-only observations, 2026-09-07 around 23:00 UTC

| Component | Observed |
|---|---|
| Host | excalibur, SSH as jordanculver; `sudo -n true` succeeds |
| CPU | 2 × Xeon E5-2680 v3, 12 physical cores/socket, 48 logical CPUs total, AVX2/FMA |
| NUMA | node 0: CPUs 0–11,24–35; node 1: 12–23,36–47; distance 10 local/21 remote |
| RAM | 503 GiB visible, approximately 499 GiB available, swap unused at discovery |
| GPU 0 | Tesla P40, 24576 MiB, PCI 04:00.0, UUID GPU-739e3766-5bad-6427-c8fc-2115ae12de31 |
| GPU 1 | Tesla P40, 24576 MiB, PCI 0d:00.0, UUID GPU-9ce90c96-486c-0dc3-8a9f-766ed9978008 |
| Topology | Both GPU host affinity NUMA0; GPU pair PHB; no NVLink |
| PCIe | Both capable Gen3 x16; idle observed Gen1 x16 (2.5 GT/s). Idle downshift is not proof of a bad link; measure during transfers later |
| Idle GPU state | 30–32°C, approximately 10 W each, P8, 0 MiB application memory, no compute applications |
| GPU power range | 125–250 W; current/default 250 W; no changes made |
| Temperature firmware thresholds | Slowdown 92°C, shutdown 95°C; memory temperature unsupported. These are NOT project operating targets |
| AI storage | `/dev/sda1`, Intel SSDSC2BB48 family SATA SSD, 440 GiB filesystem, about 184 GiB free |
| Other storage | 465.8 GiB SK Hynix SATA system disk; 465.8 GiB ST500LT012 HDD mounted `/mnt/ai-hdd` |
| Driver | 580.173.02; nvidia-smi reports CUDA compatibility 13.0 |
| Toolkit | `/usr/bin/nvcc` 12.0.140; binary links `libcudart.so.12` |
| Compiled GPU target | `cuobjdump --list-elf c/qwen36` shows `sm_61.cubin` |
| Profilers found | Nsight Systems 2022.4.2; nvprof 12.0.146. Their successful launch/trace collection is NOT yet tested |
| Runtime device attributes | 30 SMs/device, 3 MiB L2/device, 48 KiB shared/block, 96 KiB shared/SM, 64K registers/SM, 2 async engines, peer access eligible in both directions; [raw query](results/2026-09-07-device-attributes.json) |

Driver-supported CUDA version is not the installed compiler. Preserve the functioning CUDA 12 toolchain; do not use Colibrì's `CUDA_ARCH=portable` bundle, which targets much newer GPUs.

## Cooling observation that must be resolved before stress

`ipmitool sensor` reported FAN3 500 RPM critical. A later `sdr type fan` reported 2100 RPM okay; `sensor get FAN3` then reported 400 RPM Lower Critical. The SEL around 23:01–23:02 UTC repeatedly asserts/deasserts lower-critical events for sensor 0x42 (FAN2) and 0x43 (FAN3). Several IPMI calls also emitted unexpected-response-ID messages. This is evidence of unstable readings/control/airflow, not proof of a particular mechanical failure. GPUs being cool while idle does not resolve it.

Read-only follow-up should sample sensors serially (one IPMI session at a time), correlate SEL timestamps, identify fan headers and physical modules from the server documentation, and inspect physical airflow. Do not clear SEL, suppress thresholds, or apply guessed raw BMC fan commands. The user may need to inspect/reseat/replace fans or verify a sensor/control issue.

### Policy correction, 2026-09-08

The primary identified cause was the active custom monitor's intentionally quiet configuration: it held both BMC fan zones at 20% PWM when GPUs were cool. Its full-speed policy is now 100% PWM in both zones at idle, load, and emergency temperatures; it also serializes daemon/request-hook write cycles through a lock. The post-restart status was `cap:100`, manual/full BMC mode, 100/100 zone PWM, all FAN1–FAN8 sensors `ok`, and idle P40 temperatures of 32–33°C. The [configuration record](results/2026-09-08-fan-policy.md) has exact files, verification and backup suffix.

This clears the known software under-PWM cause, not the full thermal gate. The BMC had prior intermittent raw-response warnings, and the corrected policy has not yet completed the five-minute observation, physical airflow inspection, or guarded load-canary requirements below.

### T00 idle-observation result, 2026-09-08

The five-minute idle observation is now recorded in
[T00 thermal investigation](results/T00-thermal-investigation.md). Six serial
samples over 340 seconds found both P40s at 34–35°C, zero application VRAM,
and all eight fans at 2000–2100 RPM, with no new SEL record after the corrected
service started. This passes the no-load portion of the gate only. The first
one-GPU, 125 W canaries and their five-minute cooldowns remain mandatory before
any model run or long comparison.

## Proposed conservative test policy

These thresholds are project choices, not NVIDIA specifications. Store them in versioned benchmark configuration; never silently raise them to get a run to pass.

1. Before any workload: exclusive benchmark lock, no unrelated GPU jobs, functioning independent watchdog, current sensor/device mapping, no active/repeating critical fan alarms during a 5-minute observation, and a resolved explanation for the captured oscillation.
2. Start only after both GPUs are ≤40°C and stable for 60 seconds. Sample GPU telemetry every second; stop if it is stale >3 seconds. Poll BMC serially every5 seconds with its own15-second stale limit. Abort if a required field disappears, a GPU/Xid fault occurs, a critical fan alarm appears, or either GPU reaches65°C. Also stop at ≥55°C if its10-second slope is >0.5°C/s. Monitor both GPUs even when only one is selected.
3. First canaries: short 1-second bursts, then 5 and 10 seconds, one GPU at a time, 64–256 MiB allocations, ≤1 GiB total test allocation. Cool between tests. Advance only after clean stop/cleanup and stable temperatures.
4. Initial stress canaries may temporarily cap each selected GPU at 125 W (observed supported minimum). Save and restore original limits with an independent cleanup path. Benchmark comparisons must use the same limit; label results power-limited. No overclocking or automatic jump to 250 W.
5. Run no 512-token comparison until a guarded short model canary and a longer thermal plateau have succeeded. A 512-token run at the historical 1.5 tok/s lasts roughly six minutes before startup; CPU baseline can take much longer.
6. Watchdog owns the child process group/cgroup, not an arbitrary PID name. TERM, bounded grace, then KILL only that job if required. Supervisor must survive SSH loss and continue cleanup/power restoration/cooldown. Do not rely on engine CANCEL: Qwen's serial serve loop cannot process it during generation.
7. After process exit, continue monitoring at least 5 minutes and until temperature returns to ≤40°C without rising trend; the user reported post-run heat soak. Verify VRAM/process release. Failed cooldown blocks the next run.

CPU-heavy builds/training also heat the chassis. Use bounded compilation (initially `-j2`) and monitor CPU/system sensors; no training during unresolved cooling.

## Hardware model to verify with microbenchmarks

Theoretical P40: sm_61, 30 SMs ×128 FP32 CUDA cores =3840; approximately 12 TFLOP/s FP32, approximately 0.1875 TFLOP/s FP16 arithmetic (1/64 FP32 on this GP102 class), about 47 TOPS INT8 using DP4A, 24 GiB VRAM, 346 GB/s peak GDDR5 bandwidth. DP4A packs four signed/unsigned INT8 products into an INT32 accumulation; it is not an INT4 or Tensor-Core operation. FLOP/TOPS counts use multiply+add as two operations. Verify clocks, not just SKU maximums.

The runtime query confirms 3 MiB L2 on each installed P40; the exact L1 configuration remains unverified. Do not choose a tile from an assumed L1/cache policy. Pascal tuning documents96 KiB shared memory per GP104-class SM,48 KiB per block,64K32-bit registers/SM,64 resident warps/SM and32-thread warps. Query supported attributes on these GP102 devices before relying on them; do not import P100/GP100 values. Test `__ldg`, warp shuffles, coalescing, register pressure and bank conflicts. Pascal has no independent thread scheduling, `cp.async`, BF16/FP8 Tensor-Core arithmetic, or native structured sparsity acceleration.

Use FP16 as a storage option with conversion to FP32 separately from true FP16 arithmetic. Use DP4A with bounded accumulation intervals (all signed -128 products can overflow INT32 at 131072 products); output rounding/scaling and activation quantization belong in the measured operation. FP32 accumulation and nonlinearities remain valuable.

PCIe Gen3 x16 theoretical one-way payload line ceiling is about 15.75 GB/s after 128b/130b coding, before protocol/software overhead. Actual H2D/D2H, simultaneous bidirectional and peer transfer bandwidth/latency must be measured, especially across the PHB. Four cards do not imply one 96 GiB address pool, fourfold bandwidth, or symmetric future topology.

Primary specifications, toolchain evidence and remaining gaps are linked in [papers.md](papers.md). Device attributes, P2P eligibility, actual bandwidth, loaded link speed, FP16/INT8 performance, energy and long-duration thermal stability are still unmeasured.
