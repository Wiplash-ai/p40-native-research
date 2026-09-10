#!/usr/bin/env python3
"""Forced-command identity for exact Qwen DeltaNet-state first touch."""
from __future__ import annotations

import importlib.util
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t40_base", SourceFileLoader("p40_t40_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t40-qwen-dn-firsttouch"
RESULTS_ORIGINAL_COMMAND = "p40-t40-qwen-dn-firsttouch-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t40-dn-firsttouch/c/qwen36")
ENGINE_SHA256 = "696f07b97a09b61022811d6fa0b6a4eb6a57b0be9a570c5ff70b5784c4407648"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t40-qwen-dn-firsttouch")
PROFILE_ID = "t40-exact-qwen-dn-firsttouch-64"
SCHEMA_VERSION = "p40-t40-qwen-dn-firsttouch-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
EXPECTED_HEADS = 30 * 32
FIRSTTOUCH_PATTERN = re.compile(
    rb"\[dn-firsttouch\] active: head pages node0=([0-9]+) node1=([0-9]+) unknown=([0-9]+)"
)


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_DN_PAIR=1":
            result.append("COLI_DN_FIRSTTOUCH=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        marker = FIRSTTOUCH_PATTERN.search(
            Path(result.get("stderr", {}).get("path", "")).read_bytes()
        )
        if not marker:
            result["failure_reason"] = "deltanet_firsttouch_marker_missing"
            result["status"] = "fail"
        else:
            node0, node1, unknown = (int(value) for value in marker.groups())
            result["deltanet_firsttouch"] = {
                "node0_heads": node0,
                "node1_heads": node1,
                "unknown_heads": unknown,
            }
            if node0 + node1 + unknown != EXPECTED_HEADS or not node0 or not node1 or unknown:
                result["failure_reason"] = "deltanet_firsttouch_node_placement_invalid"
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
