#!/usr/bin/env python3
"""T03-only parsers and validators for proposed Colibri timing instrumentation.

This module intentionally has no CUDA, SSH, subprocess, or engine-source dependency.
It validates the line-oriented contract that a later isolated experimental patch must
emit from direct qwen36 execution.
"""
from __future__ import annotations

import json
from collections import defaultdict

TIMER_PREFIX = "COLI_T03_TIMER "
COUNTER_PREFIX = "COLI_T03_COUNTER "
VALID_UNITS = {"ns", "bytes", "count"}
VALID_RELATIONS = {"none", "subset", "overlap"}


def _object(line: str, prefix: str) -> dict:
    if not line.startswith(prefix):
        raise ValueError(f"expected {prefix.strip()} record")
    value = json.loads(line[len(prefix):])
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    return value


def parse_timer(line: str) -> dict:
    """Validate one duration record; durations are integer nanoseconds."""
    record = _object(line, TIMER_PREFIX)
    required = {"name", "duration", "unit", "relation"}
    if set(record) - {"name", "duration", "unit", "relation", "parent"} or not required <= set(record):
        raise ValueError("invalid timer fields")
    if not isinstance(record["name"], str) or not record["name"]:
        raise ValueError("timer name required")
    if record["unit"] != "ns" or not isinstance(record["duration"], int) or record["duration"] < 0:
        raise ValueError("timer duration must be non-negative integer ns")
    if record["relation"] not in VALID_RELATIONS:
        raise ValueError("invalid timer relation")
    if "parent" in record and (not isinstance(record["parent"], str) or not record["parent"]):
        raise ValueError("timer parent must be non-empty string")
    return record


def parse_counter(line: str) -> dict:
    """Validate an actual-event counter or transfer-byte record."""
    record = _object(line, COUNTER_PREFIX)
    required = {"name", "value", "unit"}
    if set(record) != required or not isinstance(record["name"], str) or not record["name"]:
        raise ValueError("invalid counter fields")
    if record["unit"] not in VALID_UNITS or not isinstance(record["value"], int) or record["value"] < 0:
        raise ValueError("counter value must be a non-negative integer with known unit")
    return record


def parse_direct_output(text: str) -> dict:
    """Extract only T03 records from direct execution stdout/stderr.

    Served-request output is deliberately not accepted as timing evidence because the
    installed gateway does not emit the direct qwen36 final timing report.
    """
    timers, counters = [], []
    for line in text.splitlines():
        if line.startswith(TIMER_PREFIX):
            timers.append(parse_timer(line))
        elif line.startswith(COUNTER_PREFIX):
            counters.append(parse_counter(line))
    if not timers and not counters:
        raise ValueError("no T03 direct-execution records")
    return {"timers": timers, "counters": counters}


def summarize(records: dict) -> dict:
    """Summarize additive timers only; subset/overlap intervals are never summed."""
    additive_ns = sum(item["duration"] for item in records["timers"] if item["relation"] == "none")
    counters: dict[str, int] = defaultdict(int)
    units: dict[str, str] = {}
    for item in records["counters"]:
        if item["name"] in units and units[item["name"]] != item["unit"]:
            raise ValueError("counter unit changed within report")
        units[item["name"]] = item["unit"]
        counters[item["name"]] += item["value"]
    return {"additive_duration_ns": additive_ns, "counters": dict(counters), "counter_units": units}
