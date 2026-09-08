#!/usr/bin/env python3
"""Client for the fixed-profile, forced-command T05A Q8 control."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CLIENT_TIMEOUT_SECONDS = 1200


def ssh_argv(host: str, identity: Path, *, read_results: bool = False) -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        "-i", str(identity), host, "p40-t05a-q8-results" if read_results else "p40-t05a-q8",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="jordanculver@192.168.1.194")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--read-results", action="store_true")
    args = parser.parse_args()
    try:
        completed = subprocess.run(
            ssh_argv(args.host, args.identity, read_results=args.read_results),
            input=None if args.read_results else json.dumps({"dry_run": args.dry_run}),
            text=True, capture_output=True,
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
