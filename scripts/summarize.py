#!/usr/bin/env python3
"""Validate minimum durable result fields using the standard library."""
from __future__ import annotations
import argparse, json
REQUIRED = ("schema_version", "task", "status", "run_id", "command", "snapshot", "cleanup")
def validate(data):
    missing=[key for key in REQUIRED if key not in data]
    if data.get("schema_version") != "p40-result-v1": missing.append("schema_version=p40-result-v1")
    if not isinstance(data.get("command"), list): missing.append("command:list")
    if not isinstance(data.get("cleanup"), dict): missing.append("cleanup:object")
    return missing
def main():
    p=argparse.ArgumentParser(); p.add_argument("result"); a=p.parse_args(); data=json.load(open(a.result, encoding="utf-8")); missing=validate(data)
    print(json.dumps({"valid":not missing,"missing":missing,"status":data.get("status")}, sort_keys=True)); return 0 if not missing else 1
if __name__ == "__main__": raise SystemExit(main())
