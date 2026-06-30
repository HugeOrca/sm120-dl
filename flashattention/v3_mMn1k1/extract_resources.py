#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "pybind" / "build" / "build.log"
DEFAULT_OUTPUT = ROOT / "pybind" / "build" / "resource.log"

FUNC_RE = re.compile(r"Compiling entry function '([^']*flash_atten_v3_fwd[^']*)'")
STACK_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
USED_RE = re.compile(r"Used (\d+) registers")
CAUSAL_RE = re.compile(r"ELb([01])ELi")
TRAITS_RE = re.compile(
    r"Flash_fwd_kernel_traitsILi(?P<head_num>\d+)ELi(?P<head_dim>\d+)"
    r"ELi(?P<block_m>\d+)ELi(?P<block_n>\d+)ELi(?P<nwarps>\d+)ELi(?P<stage>\d+)E"
)


def readable_func_name(func_name):
    causal_match = CAUSAL_RE.search(func_name)
    traits_match = TRAITS_RE.search(func_name)
    if not causal_match or not traits_match:
        return "", func_name

    values = {key: int(value) for key, value in traits_match.groupdict().items()}
    kernel_func_name = (
        f"hd{values['head_dim']}_bm{values['block_m']}_bn{values['block_n']}"
        f"_w{values['nwarps']}_s{values['stage']}"
    )
    return f"casual{causal_match.group(1)}", kernel_func_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    current_func = None
    current_stack = None
    current_spill = None
    rows = []

    for line in args.input.read_text(errors="replace").splitlines():
        match = FUNC_RE.search(line)
        if match:
            current_func = readable_func_name(match.group(1))
            current_stack = None
            current_spill = None
            continue

        if current_func is None:
            continue

        match = STACK_RE.search(line)
        if match:
            current_stack = int(match.group(1))
            spill_stores = int(match.group(2))
            spill_loads = int(match.group(3))
            current_spill = spill_stores + spill_loads
            continue

        match = USED_RE.search(line)
        if match and current_stack is not None and current_spill is not None:
            registers = int(match.group(1))
            casual, kernel_func_name = current_func
            rows.append(f"{casual} {kernel_func_name} {current_spill} {current_stack} {registers}")
            current_func = None
            current_stack = None
            current_spill = None

    header = "casual kernel_func_name spill stack registers"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
