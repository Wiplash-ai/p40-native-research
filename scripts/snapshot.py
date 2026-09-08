#!/usr/bin/env python3
"""Collect a credential-free, deterministic local benchmark snapshot."""
from __future__ import annotations
import argparse, hashlib, json, os, platform
from datetime import datetime, timezone

ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "CUDA_VISIBLE_DEVICES")

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def cpu_identity() -> dict:
    model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"architecture": platform.machine(), "model": model, "logical_cpus": os.cpu_count()}

def collect(binary: str | None = None, model: str | None = None) -> dict:
    data = {"captured_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
            "cpu": cpu_identity(), "environment": {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ},
            "gpu": {"identity": "not_queried", "reason": "T01 avoids CUDA initialization"}}
    for key, path in (("binary", binary), ("model", model)):
        if path:
            data[key] = {"path": os.path.abspath(path), "sha256": sha256(path)}
    return data

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--binary"); parser.add_argument("--model"); parser.add_argument("--output")
    args = parser.parse_args(); result = collect(args.binary, args.model); text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output: open(args.output, "w", encoding="utf-8").write(text)
    else: print(text, end="")
    return 0
if __name__ == "__main__": raise SystemExit(main())
