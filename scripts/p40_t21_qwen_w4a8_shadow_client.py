#!/usr/bin/env python3
"""Client for the fixed T21 exact-output W4A8 shadow canary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="jordanculver@192.168.1.194")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--read-results", action="store_true")
    args = parser.parse_args()
    command = "p40-t21-qwen-w4a8-shadow-results" if args.read_results else "p40-t21-qwen-w4a8-shadow"
    try:
        run = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15", "-i", str(args.identity), args.host, command],
            input=None if args.read_results else json.dumps({"dry_run": args.dry_run}), text=True, capture_output=True,
            timeout=30 if args.read_results else 2400)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "client_failure", "error": type(error).__name__}), file=sys.stderr); return 1
    print(run.stdout, end="")
    print(run.stderr, end="", file=sys.stderr)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
