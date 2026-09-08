#!/usr/bin/env python3
"""A separately forced, fixed 256-output Qwen thermal-plateau profile."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-qwen-canary-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-qwen-canary-guard.py")
SPEC = importlib.util.spec_from_loader("p40_qwen_control_base", SourceFileLoader("p40_qwen_control_base", str(SOURCE)))
assert SPEC and SPEC.loader
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)

OUTPUT_TOKENS = 256
EXPECTED_ORIGINAL_COMMAND = "p40-qwen-control-256"
RESULTS_ORIGINAL_COMMAND = "p40-qwen-control-256-results"
PROFILE_ID = "control-256"
SCHEMA_VERSION = "p40-qwen-control-256-v1"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/qwen-controls-256")

qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
qwen.PROFILE_ID = PROFILE_ID
qwen.SCHEMA_VERSION = SCHEMA_VERSION
qwen.RESULT_DIRECTORY = RESULT_DIRECTORY


def main() -> int:
    return qwen.main()


if __name__ == "__main__":
    raise SystemExit(main())
