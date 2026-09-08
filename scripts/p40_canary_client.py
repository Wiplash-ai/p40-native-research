#!/usr/bin/env python3
"""Client for the forced-command P40 primitive-canary executor."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def request_from_args(args: argparse.Namespace) -> dict:
    return {
        "primitive": args.primitive,
        "gpu": args.gpu,
        "duration_seconds": args.duration,
        "bytes": args.bytes,
        "memory_cap_mib": args.memory_cap_mib,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }


def ssh_argv(host: str, identity: Path) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15", "-i", str(identity), host, "p40-canary"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="jordanculver@192.168.1.194")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--primitive", choices=("P01-copy", "P03-dp4a-gemv"), default="P01-copy")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--duration", type=int, choices=(1, 5, 10), default=1)
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--memory-cap-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--read-results", action="store_true")
    args = parser.parse_args()
    if args.read_results:
        completed = subprocess.run([*ssh_argv(args.host, args.identity)[:-1], "p40-canary-results"], text=True, capture_output=True, timeout=30)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    completed = subprocess.run(
        ssh_argv(args.host, args.identity), input=json.dumps(request_from_args(args)), text=True, capture_output=True, timeout=380
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
