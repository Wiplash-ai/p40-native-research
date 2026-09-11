#!/usr/bin/env python3
"""Stop the local Colibrì service if a P40 reaches its configured limit."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


LIMIT_C = int(os.environ.get("COLIBRI_THERMAL_C", "70"))
POLL_SECONDS = float(os.environ.get("COLIBRI_THERMAL_POLL_SECONDS", "5"))
STOP_SECONDS = float(os.environ.get("COLIBRI_THERMAL_STOP_SECONDS", "45"))
NVIDIA_SMI = os.environ.get("COLIBRI_NVIDIA_SMI", "/usr/bin/nvidia-smi")


def hottest_c() -> int:
    output = subprocess.check_output(
        [NVIDIA_SMI, "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no temperatures")
    return max(values)


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("thermal supervisor requires a child command")
    if LIMIT_C < 1 or POLL_SECONDS <= 0 or STOP_SECONDS <= 0:
        raise SystemExit("invalid thermal supervisor configuration")
    try:
        hottest = hottest_c()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"[thermal-supervisor] startup telemetry unavailable: {error}", file=sys.stderr,
              flush=True)
        # Do not start a GPU-resident model when thermal telemetry is unknown.
        return 0
    if hottest >= LIMIT_C:
        print(f"[thermal-supervisor] {hottest} C >= {LIMIT_C} C; refusing service start",
              file=sys.stderr, flush=True)
        return 0
    child = subprocess.Popen(argv)
    while child.poll() is None:
        try:
            hottest = hottest_c()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"[thermal-supervisor] telemetry unavailable: {error}", file=sys.stderr,
                  flush=True)
        else:
            if hottest >= LIMIT_C:
                print(f"[thermal-supervisor] {hottest} C >= {LIMIT_C} C; stopping service",
                      file=sys.stderr, flush=True)
                child.send_signal(signal.SIGTERM)
                deadline = time.monotonic() + STOP_SECONDS
                while child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.25)
                if child.poll() is None:
                    print("[thermal-supervisor] child did not exit; killing", file=sys.stderr,
                          flush=True)
                    child.kill()
                    child.wait()
                # A thermal stop is deliberate. Do not trigger systemd restart
                # into a still-hot chassis.
                return 0
        time.sleep(POLL_SECONDS)
    return child.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
