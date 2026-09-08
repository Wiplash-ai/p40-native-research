#!/usr/bin/env python3
"""Print explicit benchmark invocations without inheriting auto-tune flags."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def cases():
    return [{"id":"P01-copy-64m-gpu0", "primitive":"P01", "gpu":"0", "shape":"64MiB", "seed":1, "duration_seconds":1, "memory_cap_mib":256, "argv":["./benchmarks/p40_bench","--primitive","P01","--gpu","0","--bytes","67108864","--seed","1","--duration","1","--memory-cap-mib","256"]}]
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--config", default="research/config/t01-dry-run.json"); args=parser.parse_args()
    config={"mode":"dry_run", "cuda_initialized":False, "environment":{"CUDA_VISIBLE_DEVICES":"explicit-per-case", "COLI_CUDA":"unset", "HEAT_FILE":"unset"}, "cases":cases()}
    if not args.dry_run: parser.error("T01 supports --dry-run only")
    print(json.dumps(config, indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
