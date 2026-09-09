#!/usr/bin/env python3
"""Forced-command guard for the bounded T11 attention projection control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t11_base", SourceFileLoader("p40_t11_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05
SPEC.loader.exec_module(t05)

t05.EXPECTED_ORIGINAL_COMMAND = "p40-t11-attention-cpuorder"
t05.RESULTS_ORIGINAL_COMMAND = "p40-t11-attention-cpuorder-results"
t05.BENCHMARK = Path("/home/jordanculver/p40-native-research/t11-attention-bench/qwen_attention_cpuorder_control")
t05.BENCHMARK_SHA256 = "b7d74121b78bf7a7003c83eea88dcacf68754e5de27f122b470e4e83f9427ca9"
t05.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t11-attention-cpuorder-controls")
t05.PROFILE_ID = "t11-attention-projections-cpuorder-10x"
t05.SCHEMA_VERSION = "p40-t11-attention-cpuorder-q8-v1"
t05.WATCHDOG_SECONDS = 180
t05.FIXED_ARGUMENTS = (
    "--profile", "attention-projections-cpuorder-10x-2048-8192-512-4096-2048", "--gpu", "0",
    "--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "288", "--seed", "1",
)


def main() -> int:
    return t05.main()


if __name__ == "__main__":
    raise SystemExit(main())
