#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent / "pybind" / "build"))
import flashatten_v3_sm120


def parse_bool(value):
    if value in ("1", "true", "True", "yes", "y"):
        return True
    if value in ("0", "false", "False", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def list_configs(names_only):
    for config in flashatten_v3_sm120.configs():
        name, head_dim, block_m, block_n, nwarps, stage, smem_bytes = config
        if names_only:
            print(name)
        else:
            print(
                f"{name} hd={head_dim} bm={block_m} bn={block_n} "
                f"nwarps={nwarps} stage={stage} smem={smem_bytes}"
            )


def select_config(name):
    for config in flashatten_v3_sm120.configs():
        if config[0] == name:
            return config
    available = ", ".join(config[0] for config in flashatten_v3_sm120.configs())
    raise SystemExit(f"unknown --func '{name}'. available: {available}")


def validate_shape(config, seq_len, head_dim):
    name, config_head_dim, block_m, block_n, _, _, _ = config
    if head_dim != config_head_dim:
        raise SystemExit(f"{name} expects head_dim={config_head_dim}, got {head_dim}")
    if seq_len % block_m != 0:
        raise SystemExit(f"{name} expects seq_len to be a multiple of block_m={block_m}, got {seq_len}")
    if seq_len % block_n != 0:
        raise SystemExit(f"{name} expects seq_len to be a multiple of block_n={block_n}, got {seq_len}")


def make_inputs(seq_len, head_dim, head_num, batch, causal):
    torch.manual_seed(20260629 + seq_len + head_dim + head_num + batch + int(causal))
    q = (torch.randn(batch, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16) * 0.5).contiguous()
    k = (torch.randn_like(q) * 0.5).contiguous()
    v = (torch.randn_like(q) * 0.5).contiguous()
    return q, k, v


def profiler_start():
    result = torch.cuda.cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with code {result}")


def profiler_stop():
    result = torch.cuda.cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with code {result}")


def run_once(func_name, q, k, v, causal):
    profiler_start()
    try:
        out = flashatten_v3_sm120.fwd_config(func_name, q, k, v, causal)
        torch.cuda.synchronize()
    finally:
        profiler_stop()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", help="Generated flash_fwd function config name")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-num", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--causal", type=parse_bool, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-names", action="store_true")
    args = parser.parse_args()

    if args.list or args.list_names:
        list_configs(args.list_names)
        return 0
    if not args.func:
        parser.error("--func is required unless --list or --list-names is used")
    if args.seq_len <= 0 or args.head_num <= 0 or args.batch <= 0:
        parser.error("--seq-len, --head-num, and --batch must be positive")
    if args.warmup < 0 or args.sleep < 0:
        parser.error("--warmup and --sleep must be non-negative")

    config = select_config(args.func)
    _, head_dim, _, _, _, _, _ = config
    validate_shape(config, args.seq_len, head_dim)

    with torch.no_grad():
        q, k, v = make_inputs(args.seq_len, head_dim, args.head_num, args.batch, args.causal)
        for _ in range(args.warmup):
            flashatten_v3_sm120.fwd_config(args.func, q, k, v, args.causal)
        torch.cuda.synchronize()

        print(
            f"PROFILE func={args.func} seq_len={args.seq_len} head_dim={head_dim} "
            f"head_num={args.head_num} batch={args.batch} causal={int(args.causal)}",
            flush=True,
        )
        out = run_once(args.func, q, k, v, args.causal)
        print(f"DONE func={args.func} output_shape={tuple(out.shape)}", flush=True)

    if args.sleep > 0:
        time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
