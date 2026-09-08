# Persistent full-speed fan policy

Time: 2026-09-08 04:12–04:14 UTC. Host: `excalibur`. This is an operational configuration record, not a thermal-load benchmark.

## Cause found

`wiplash-gpu-fan-monitor.service` was intentionally applying its prior quiet-office policy:

- idle cap: 20% PWM on both BMC zones;
- load mode: 60% PWM at or above 60°C GPU temperature;
- emergency mode: 100% PWM at or above 82°C.

The source was `/home/jordanculver/code/wiplash-gpu-fan-guard`, installed at `/usr/local/sbin/wiplash-gpu-fan-guard`; its service override was `/etc/systemd/system/wiplash-gpu-fan-monitor.service.d/override.conf`.

## Change

- Set `WIPLASH_GPU_FAN_CAP_PWM=100`, `WIPLASH_GPU_FAN_LOAD_PWM=100`, and retained `WIPLASH_GPU_FAN_FULL_PWM=100` with `CAP_MODE=manual`. Both zones therefore remain at 100% PWM at idle, load, and emergency temperatures.
- Added a `flock`-guarded, write-capable check so the monitor daemon and request hooks cannot issue concurrent in-band IPMI read/decide/write cycles. The script had a declared lock path but did not previously use it.
- Updated server-side operator notes. Timestamped backups use suffix `20260908T041241Z`.
- Syntax-checked the script, verified the systemd unit, reloaded systemd, and restarted only `wiplash-gpu-fan-monitor.service`.

## Verification after restart

| Check | Result |
|---|---|
| Monitor service | active |
| Requested state | `cap:100` |
| BMC mode | manual/full register (`1`) |
| BMC zone 0 / zone 1 | 100 / 100 PWM |
| Fan sensors | FAN1–FAN8 `ok`, 2000–2100 RPM on the sampled board reading |
| GPU state | both P40s idle, 32–33°C, about 10 W, zero application VRAM |
| Script copies | source and installed SHA-256 matched |

`ipmitool` had previously emitted occasional "unexpected ID" responses, including during read-only inspection. The persistent policy and serialized writes remove the known two-writer race, but this is not evidence that every BMC transport warning has disappeared.

## Impact on the research gate

This fixes the known software configuration that was reducing chassis airflow. It does **not** replace T00's five-minute idle observation, physical airflow inspection, or guarded load canaries. Do not regard it as evidence that the P40s can sustain an inference or training workload until those acceptance checks pass.
