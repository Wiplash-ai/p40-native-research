#!/usr/bin/env python3
"""Client for the fixed-profile, forced-command Qwen research guards."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "canary-16": ("p40-qwen-canary", "p40-qwen-canary-results"),
    "control-64": ("p40-qwen-control-64", "p40-qwen-control-64-results"),
    "control-256": ("p40-qwen-control-256", "p40-qwen-control-256-results"),
}
# 1,800 s server watchdog plus mandatory 300 s cooldown and SSH slack. The
# server, not this client, owns termination and recovery.
CLIENT_TIMEOUT_SECONDS = 2400


def ssh_argv(host: str, identity: Path, profile: str, *, read_results: bool = False) -> list[str]:
    command = PROFILES[profile][1 if read_results else 0]
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        "-i", str(identity), host, command,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="jordanculver@192.168.1.194")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="control-64")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--read-results", action="store_true")
    args = parser.parse_args()
    argv = ssh_argv(args.host, args.identity, args.profile, read_results=args.read_results)
    try:
        completed = subprocess.run(
            argv,
            input=None if args.read_results else json.dumps({"dry_run": args.dry_run}),
            text=True,
            capture_output=True,
            timeout=30 if args.read_results else CLIENT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "client_failure", "error": type(error).__name__}), file=sys.stderr)
        return 1
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
