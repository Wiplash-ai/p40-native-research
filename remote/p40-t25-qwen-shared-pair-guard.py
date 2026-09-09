#!/usr/bin/env python3
"""Forced-command identity for the exact T25 shared-MLP pair canary."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t23-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t23-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t25_base", SourceFileLoader("p40_t25_base", str(SOURCE)))
assert SPEC and SPEC.loader
t23 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t23
SPEC.loader.exec_module(t23)

EXPECTED_ORIGINAL_COMMAND = "p40-t25-qwen-shared-pair"
RESULTS_ORIGINAL_COMMAND = "p40-t25-qwen-shared-pair-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t25-shared-pair/c/qwen36")
ENGINE_SHA256 = "a9dac675ea2f37e1ba9ffb53353400e0d6e3127d035fddeecb91b17794194104"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t25-qwen-shared-pair")
PROFILE_ID = "t25-exact-qwen-shared-pair-16"
SCHEMA_VERSION = "p40-t25-qwen-shared-pair-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
SHARED_PAIR_MARKER = b"[dn-cpuorder] exact Q8 shared gate/up pair issue/join active; scalar gate overlap enabled"


_base_model_argv = t23.model_argv
_base_run = t23.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_SHARED_CPUORDER=1":
            result.append("COLI_CUDA_SHARED_PAIR=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if SHARED_PAIR_MARKER not in stderr:
            result["failure_reason"] = "shared_pair_not_active"
            result["status"] = "fail"
    return result


def configure(module) -> None:
    module.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
    module.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
    module.ENGINE = ENGINE
    module.ENGINE_SHA256 = ENGINE_SHA256
    module.RESULT_DIRECTORY = RESULT_DIRECTORY
    module.PROFILE_ID = PROFILE_ID
    module.SCHEMA_VERSION = SCHEMA_VERSION
    module.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
    module.model_argv = model_argv


configure(t23)
configure(t23.t15)
configure(t23.t15.t12)
configure(t23.t15.t12.t09)
configure(t23.t15.t12.t09.t06)
configure(t23.t15.t12.t09.t06.qwen)
t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = 16
t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
