#!/usr/bin/env python3
"""Forced-command identity for the opt-in T06 Qwen DeltaNet integration."""
from __future__ import annotations

import datetime as dt
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import sys
import uuid
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-qwen-canary-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-qwen-canary-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t06_qwen_base", SourceFileLoader("p40_t06_qwen_base", str(SOURCE)))
assert SPEC and SPEC.loader
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)

EXPECTED_ORIGINAL_COMMAND = "p40-t06-qwen-dn-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t06-qwen-dn-cpuorder-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t06/c/qwen36")
ENGINE_SHA256 = "c515e657d3e999f7422aa9a0fa1911dcca017ac7ebd840d993ef3ea533183b46"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t06-qwen-dn-cpuorder")
PROFILE_ID = "t06-dn-cpuorder-16"
SCHEMA_VERSION = "p40-t06-qwen-dn-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"


def engine_digest() -> str:
    return hashlib.sha256(ENGINE.read_bytes()).hexdigest()


def model_argv() -> list[str]:
    environment = {
        "COLI_CUDA": "1", "COLI_GPUS": "0,1", "CUDA_EXPERT_GB": "auto",
        "COLI_DENSE_I8": "1", "COLI_TIMERS": "1", "COLI_CUDA_PROFILE": "1",
        "COLI_CUDA_DN_CPUORDER": "1", "COLI_CUDA_DN_DEVICE": "0",
        "OMP_NUM_THREADS": "24", "OMP_DYNAMIC": "FALSE", "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_PROC_BIND": "close", "OMP_PLACES": "cores", "Q36_MAXT": "8192",
        "PILOT": "0", "HOT": "0", "COLIBRI_RESIDENT": "0", "SNAP": str(qwen.MODEL),
        "N_NEW": "16", "NOSTREAM": "1",
    }
    return ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin",
            *[f"{key}={value}" for key, value in environment.items()],
            str(ENGINE), "256", "4", str(qwen.PROMPT)]


_base_run = qwen.run


def cuda_runtime_diagnostic(stderr: bytes) -> bool:
    """Reject runtime failures, while permitting the normal device inventory."""
    if b"[dn-cpuorder] CUDA failure" in stderr:
        return True
    for line in stderr.splitlines():
        if line.startswith(b"[CUDA] ") and not line.startswith(b"[CUDA] device "):
            return True
    return False


def run(request: dict) -> dict:
    if not request["dry_run"] and (not ENGINE.is_file() or not ENGINE_SHA256 or engine_digest() != ENGINE_SHA256):
        return {
            "schema_version": SCHEMA_VERSION, "profile_id": PROFILE_ID,
            "run_id": str(uuid.uuid4()), "request": request,
            "command": model_argv(), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "fail",
            "failure_reason": "pinned_engine_missing_or_changed", "telemetry": [],
            "cleanup": {"actions": [], "power_restored": False}, "cuda_initialized": False,
        }
    result = _base_run(request)
    if not request["dry_run"]:
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if result.get("stdout", {}).get("sha256") != EXPECTED_STDOUT_SHA256:
            result["failure_reason"] = "generated_output_hash_mismatch"
            result["status"] = "fail"
        elif b"[dn-cpuorder] exact Q8 cache active" not in stderr:
            result["failure_reason"] = "dn_cpuorder_not_active"
            result["status"] = "fail"
        elif cuda_runtime_diagnostic(stderr):
            result["failure_reason"] = "cuda_runtime_diagnostic"
            result["status"] = "fail"
    return result


qwen.ENGINE = ENGINE
qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
qwen.PROFILE_ID = PROFILE_ID
qwen.SCHEMA_VERSION = SCHEMA_VERSION
qwen.OUTPUT_TOKENS = 16
qwen.model_argv = model_argv
qwen.run = run


def main() -> int:
    return qwen.main()


if __name__ == "__main__":
    raise SystemExit(main())
