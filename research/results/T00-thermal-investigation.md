# T00 — thermal and fan-gate investigation

Date: 2026-09-08.  Status: **pass for the idle-observation sub-gate only**.

This result does not claim a sustained-load thermal plateau.  It authorizes
only T02's separately guarded, one-GPU 1-second canaries after the benchmark
binary and watchdog integration exist.  It does not authorize a Qwen model
load, a 512-token run, training, or a power-limit change.

## Cause and remediation evidence

The previously observed FAN2/FAN3 alarms ended at SEL record `200a`,
2026-09-08 04:12:39 UTC.  The active fan daemon was started at 04:12:42 UTC
with the persisted full-PWM override.  Its source, systemd drop-in, and BMC
status show the intended policy: manual/full BMC mode, both PWM zones at 100%,
and state `cap:100`.  The earlier quiet 20% idle-cap policy was therefore the
identified software cause; no thresholds were suppressed and SEL was not
cleared.

## Idle observation

The raw six-sample capture is
[`raw/T00-idle-telemetry-20260908T0518Z.log`](raw/T00-idle-telemetry-20260908T0518Z.log)
(`SHA-256 295bd59ad0bee397447ef808ae9f78affe088fd2e061f2b279fd54ef5f1f575d`).
It spans 05:17:43–05:23:23 UTC (340 seconds), with serial IPMI reads and no
GPU/model workload.

| Signal | Observed range / result |
|---|---|
| GPU temperature | GPU0 34°C; GPU1 35°C in every sample |
| GPU state | P8, 0% utilization, 0 MiB application VRAM |
| GPU power | 10.05–11.53 W |
| FAN1–FAN8 | 2000–2100 RPM in every sample |
| New critical FAN/thermal SEL | none; newest record remained `200a` |
| BMC at final check | manual/full (`1`), zone0 100%, zone1 100% |

The capture has one benign collection-format defect: `/bin/sh` interpreted the
separator beginning with `---` as a `printf` option.  It did not affect any
timestamp, GPU, fan, or SEL command; the raw capture retains it rather than
being rewritten.

## Independent telemetry path

A separate `p40-telemetry` SSH key now has `restrict` plus a root-owned forced
command at `/usr/local/sbin/p40-telemetry-readonly`.  It can emit only a JSON
GPU/fan/SEL snapshot.  It cannot allocate GPUs, load models, change BMC state,
open a shell, or forward ports.  The public-key fingerprint is
`SHA256:CBcuJbzlpZ4eWacBpLBzdCj6rsdLqXm2hZ1hJvnWte8`.

The allowed command returned two P40s, eight fans, `fan_critical:false`, and
`device_error:false`.  An attempted `uname -a` through that same key exited
126 with `p40 telemetry key only accepts: p40-telemetry`.

## Next safety gates

1. T02 must integrate this independent telemetry source with the existing
   process-group guard and take only 1-, 5-, then 10-second one-GPU canaries
   at the 125 W project cap.
2. Continue telemetry for five minutes after each canary; require GPU return
   to <=40°C, no rising trend, released VRAM, and no new critical SEL record.
3. Only after a clean guarded model canary and a longer thermal plateau may
   T04 load Qwen or run the fixed 512-output comparison.
