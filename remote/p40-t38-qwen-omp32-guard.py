#!/usr/bin/env python3
"""Forced-command identity for the exact Qwen 32-thread CPU control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t38_base", SourceFileLoader("p40_t38_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t38-qwen-omp32"
RESULTS_ORIGINAL_COMMAND = "p40-t38-qwen-omp32-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t23-dn-pair/c/qwen36")
ENGINE_SHA256 = "c014b491249b6692995a95f8d3e3ed5d0ae3dae657df4e5ae98e1c501dea21d7"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t38-qwen-omp32")
PROFILE_ID = "t38-exact-qwen-omp32-64"
SCHEMA_VERSION = "p40-t38-qwen-omp32-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    return ["OMP_NUM_THREADS=32" if argument == "OMP_NUM_THREADS=24" else
            f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
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
