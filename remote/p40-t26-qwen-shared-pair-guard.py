#!/usr/bin/env python3
"""Forced-command identity for the fixed 64-token T26 shared-pair comparison."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t25-qwen-shared-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t25-qwen-shared-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t26_base", SourceFileLoader("p40_t26_base", str(SOURCE)))
assert SPEC and SPEC.loader
t25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t25
SPEC.loader.exec_module(t25)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t26-qwen-shared-pair"
RESULTS_ORIGINAL_COMMAND = "p40-t26-qwen-shared-pair-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t25-shared-pair/c/qwen36")
ENGINE_SHA256 = "a9dac675ea2f37e1ba9ffb53353400e0d6e3127d035fddeecb91b17794194104"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t26-qwen-shared-pair")
PROFILE_ID = "t26-exact-qwen-shared-pair-64"
SCHEMA_VERSION = "p40-t26-qwen-shared-pair-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"


_base_model_argv = t25.model_argv
_base_run = t25.run


def model_argv() -> list[str]:
    return [f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
            for argument in _base_model_argv()]


def run(request: dict) -> dict:
    return _base_run(request)


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


configure(t25)
configure(t25.t23)
configure(t25.t23.t15)
configure(t25.t23.t15.t12)
configure(t25.t23.t15.t12.t09)
configure(t25.t23.t15.t12.t09.t06)
configure(t25.t23.t15.t12.t09.t06.qwen)
t25.t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t25.t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t25.t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
