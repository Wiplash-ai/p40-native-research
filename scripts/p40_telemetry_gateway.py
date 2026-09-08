#!/usr/bin/env python3
"""Read one P40 health snapshot through the forced-command SSH key."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class TelemetryProtocolError(ValueError):
    """The restricted server did not return a safe, complete v1 snapshot."""


def ssh_argv(host: str, identity: str | None, timeout: int) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"]
    if identity:
        argv.extend(["-o", "IdentitiesOnly=yes", "-i", identity])
    return [*argv, host, "p40-telemetry"]


def parse_snapshot(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise TelemetryProtocolError("invalid JSON") from error
    if not isinstance(data, dict) or data.get("schema_version") != "p40-telemetry-v1":
        raise TelemetryProtocolError("unexpected telemetry schema")
    if not isinstance(data.get("gpus"), list) or len(data["gpus"]) < 2:
        raise TelemetryProtocolError("missing GPU telemetry")
    if not isinstance(data.get("fans"), list) or len(data["fans"]) < 8:
        raise TelemetryProtocolError("missing fan telemetry")
    if not isinstance(data.get("device_error"), bool) or not isinstance(data.get("fan_critical"), bool):
        raise TelemetryProtocolError("invalid safety flags")
    return data


def fetch(host: str, identity: str | None = None, timeout: int = 15) -> dict:
    completed = subprocess.run(
        ssh_argv(host, identity, timeout), check=False, capture_output=True, text=True, timeout=timeout + 5
    )
    if completed.returncode:
        raise TelemetryProtocolError(f"telemetry SSH failed with exit {completed.returncode}")
    return parse_snapshot(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="jordanculver@192.168.1.194")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    try:
        snapshot = fetch(args.host, str(args.identity) if args.identity else None, args.timeout)
    except (OSError, subprocess.TimeoutExpired, TelemetryProtocolError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
