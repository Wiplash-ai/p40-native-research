#!/usr/bin/env python3
"""Forced-command identity for exact Qwen host-side expert-tier take profiling."""
from __future__ import annotations

import importlib.util
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t43_base", SourceFileLoader("p40_t43_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t43-qwen-qtier-take-profile"
RESULTS_ORIGINAL_COMMAND = "p40-t43-qwen-qtier-take-profile-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t43-qtier-take-profile/c/qwen36")
ENGINE_SHA256 = "2aae5e44ea5bbec269c649e35ea5c578cf2abbdec8e459b6b363f19bf787e193"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t43-qwen-qtier-take-profile")
PROFILE_ID = "t43-exact-qwen-qtier-take-profile-64"
SCHEMA_VERSION = "p40-t43-qwen-qtier-take-profile-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
HOST_TAKE_LINE = re.compile(
    rb"\[qtier\]\s+host-take dev ([01]): (\d+) calls \| wait ([0-9.]+) ms, "
    rb"accumulate ([0-9.]+) ms"
)


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_PROFILE=1":
            result.append("COLI_QTIER_TAKE_PROFILE=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        profiles = {int(dev): values for dev, *values in HOST_TAKE_LINE.findall(stderr)}
        if len(profiles) != 2 or any(int(values[0]) <= 0 for values in profiles.values()):
            result["failure_reason"] = "qtier_take_profile_missing_or_zero"
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


configure(t24)
configure(t24.t23)
configure(t24.t23.t15)
configure(t24.t23.t15.t12)
configure(t24.t23.t15.t12.t09)
configure(t24.t23.t15.t12.t09.t06)
configure(t24.t23.t15.t12.t09.t06.qwen)
t24.t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t24.t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t24.t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
