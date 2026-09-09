#!/usr/bin/env python3
"""Forced-command identity for the exact T22 async-expert profile."""
from __future__ import annotations

import importlib.util
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t15-qwen-shared-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t15-qwen-shared-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t22_base", SourceFileLoader("p40_t22_base", str(SOURCE)))
assert SPEC and SPEC.loader
t15 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t15
SPEC.loader.exec_module(t15)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t22-qwen-async-profile"
RESULTS_ORIGINAL_COMMAND = "p40-t22-qwen-async-profile-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t22-async-profile/c/qwen36")
ENGINE_SHA256 = "af708bad3cf1c0370f13357bce9a666852799839053770fb1a97d2cb778507b9"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t22-qwen-async-profile")
PROFILE_ID = "t22-exact-qwen-async-expert-profile-64"
SCHEMA_VERSION = "p40-t22-qwen-async-profile-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
EVENTS = re.compile(
    rb"\[qtier\] group_stats: (\d+) calls, (\d+) experts \| h2d ([0-9.]+) ms, kernel ([0-9.]+) ms, d2h ([0-9.]+) ms"
)


_base_model_argv = t15.model_argv
_base_run = t15.run


def model_argv() -> list[str]:
    return [f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
            for argument in _base_model_argv()]


def run(request: dict) -> dict:
    result = _base_run(request)
    if request["dry_run"] or result.get("status") != "pass":
        return result
    stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
    matches = list(EVENTS.finditer(stderr))
    if not matches:
        result["failure_reason"] = "async_expert_profile_missing"
        result["status"] = "fail"
        return result
    calls, experts, h2d, kernel, d2h = matches[-1].groups()
    values = [float(h2d), float(kernel), float(d2h)]
    if int(calls) < 1 or int(experts) < 1 or any(value <= 0 for value in values):
        result["failure_reason"] = "async_expert_profile_nonpositive"
        result["status"] = "fail"
        return result
    result["async_expert_profile_ms"] = {
        "calls": int(calls), "experts": int(experts), "h2d": values[0],
        "kernel": values[1], "d2h": values[2],
    }
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


configure(t15)
configure(t15.t12)
configure(t15.t12.t09)
configure(t15.t12.t09.t06)
configure(t15.t12.t09.t06.qwen)
t15.t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
