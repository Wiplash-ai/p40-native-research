#!/usr/bin/env python3
"""Forced SSH command: emit one credential-free P40 health snapshot.

This program intentionally has no arguments and never invokes a shell.  It is
installed root-owned on excalibur and is reached only by the telemetry SSH key.
The caller cannot use that key to load a model, change BMC state, or run an
arbitrary command.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys


EXPECTED_ORIGINAL_COMMAND = "p40-telemetry"
GPU_FIELDS = (
    "index,uuid,name,temperature.gpu,power.draw,power.limit,utilization.gpu,"
    "memory.used,memory.total,pstate"
)


def call(argv: list[str], errors: dict[str, str], name: str) -> str:
    try:
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=12
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        errors[name] = type(error).__name__
        return ""
    if completed.returncode:
        errors[name] = f"exit_{completed.returncode}"
        return ""
    return completed.stdout


def clean_number(value: str) -> float | None:
    token = value.strip().removesuffix(" W").removesuffix(" %").removesuffix(" MiB")
    if token in {"", "N/A", "[Not Supported]"}:
        return None
    return float(token)


def parse_gpus(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != 10:
            continue
        rows.append(
            {
                "index": int(row[0]),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "temperature_c": clean_number(row[3]),
                "power_w": clean_number(row[4]),
                "power_limit_w": clean_number(row[5]),
                "utilization_percent": clean_number(row[6]),
                "memory_used_mib": clean_number(row[7]),
                "memory_total_mib": clean_number(row[8]),
                "pstate": row[9].strip(),
            }
        )
    return rows


def parse_fans(text: str) -> list[dict[str, object]]:
    fans: list[dict[str, object]] = []
    for line in text.splitlines():
        pieces = [piece.strip() for piece in line.split("|")]
        if len(pieces) < 3 or not pieces[0].upper().startswith("FAN"):
            continue
        try:
            rpm = float(pieces[1])
        except ValueError:
            rpm = None
        fans.append({"name": pieces[0], "rpm": rpm, "unit": pieces[2]})
    return fans


def main() -> int:
    if os.environ.get("SSH_ORIGINAL_COMMAND") != EXPECTED_ORIGINAL_COMMAND:
        sys.stderr.write("p40 telemetry key only accepts: p40-telemetry\n")
        return 126

    errors: dict[str, str] = {}
    gpu_text = call(
        [
            "/usr/bin/nvidia-smi",
            f"--query-gpu={GPU_FIELDS}",
            "--format=csv,noheader",
        ],
        errors,
        "nvidia_smi",
    )
    fan_text = call(["/usr/bin/sudo", "-n", "/usr/bin/ipmitool", "sensor"], errors, "ipmi_sensor")
    sel_text = call(["/usr/bin/sudo", "-n", "/usr/bin/ipmitool", "sel", "elist"], errors, "ipmi_sel")
    gpus, fans = parse_gpus(gpu_text), parse_fans(fan_text)
    fan_events = [
        line.strip()
        for line in sel_text.splitlines()
        if "Fan FAN" in line and "Lower Critical" in line
    ][-32:]
    snapshot = {
        "schema_version": "p40-telemetry-v1",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpus": gpus,
        "fans": fans,
        "fan_critical": any(fan["rpm"] is not None and fan["rpm"] < 500 for fan in fans),
        "device_error": bool(errors) or len(gpus) < 2 or len(fans) < 8,
        "recent_fan_events": fan_events,
        "errors": errors,
    }
    print(json.dumps(snapshot, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
