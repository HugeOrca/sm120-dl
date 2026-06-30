#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent / "pybind" / "build"))
import flashatten_v3_sm120


LABEL_RE = re.compile(
    r"^b(?P<batch>\d+)_hn(?P<head_num>\d+)_seq(?P<seq_len>\d+)_hd(?P<head_dim>\d+)_(?:casual|causal)(?P<causal>[01])$"
)


def parse_bool(value):
    if value in ("1", "true", "True", "yes", "y"):
        return True
    if value in ("0", "false", "False", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def parse_input_config(values):
    if len(values) == 1:
        value = values[0]
        match = LABEL_RE.match(value)
        if match:
            return (
                int(match.group("seq_len")),
                int(match.group("head_dim")),
                int(match.group("head_num")),
                int(match.group("batch")),
                bool(int(match.group("causal"))),
            )
        if "," in value:
            values = [part.strip() for part in value.split(",") if part.strip()]

    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "--config expects either b<B>_hn<H>_seq<S>_hd<D>_casual<0|1> "
            "or: SEQ_LEN HEAD_DIM HEAD_NUM BATCH CAUSAL"
        )

    seq_len, head_dim, head_num, batch = map(int, values[:4])
    causal = parse_bool(values[4])
    return seq_len, head_dim, head_num, batch, causal


def shape_label(seq_len, head_dim, head_num, batch, causal):
    return f"b{batch}_hn{head_num}_seq{seq_len}_hd{head_dim}_casual{int(causal)}"


def select_func_config(name):
    configs = flashatten_v3_sm120.configs()
    for config in configs:
        if config[0] == name:
            return config
    available = ", ".join(config[0] for config in configs)
    raise SystemExit(f"unknown --func '{name}'. available: {available}")


def list_configs():
    for config in flashatten_v3_sm120.configs():
        name, head_dim, block_m, block_n, nwarps, stage, smem_bytes = config
        print(
            f"{name} hd={head_dim} bm={block_m} bn={block_n} "
            f"nwarps={nwarps} stage={stage} smem={smem_bytes}"
        )


def validate_case(func_config, seq_len, head_dim):
    name, func_head_dim, block_m, block_n, _, _, _ = func_config
    if head_dim != func_head_dim:
        raise SystemExit(f"{name} expects head_dim={func_head_dim}, got {head_dim}")
    if seq_len % block_m != 0:
        raise SystemExit(f"{name} expects seq_len to be a multiple of block_m={block_m}, got {seq_len}")
    if seq_len % block_n != 0:
        raise SystemExit(f"{name} expects seq_len to be a multiple of block_n={block_n}, got {seq_len}")


def make_inputs(seq_len, head_dim, head_num, batch, causal):
    torch.manual_seed(20260623 + seq_len + head_dim + head_num + batch + int(causal))
    q = (torch.randn(batch, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16) * 0.5).contiguous()
    k = (torch.randn_like(q) * 0.5).contiguous()
    v = (torch.randn_like(q) * 0.5).contiguous()
    return q, k, v


def run_custom(func_name, q, k, v, causal):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = flashatten_v3_sm120.fwd_config(func_name, q, k, v, causal)
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="+",
        metavar="INPUT",
        help="Input dimensions: b<B>_hn<H>_seq<S>_hd<D>_casual<0|1>, or SEQ_LEN HEAD_DIM HEAD_NUM BATCH CAUSAL",
    )
    parser.add_argument("--func", help="Generated flash_fwd function config name")
    args = parser.parse_args()

    if args.config is None and args.func is None:
        list_configs()
        return 0
    if args.config is None or args.func is None:
        parser.error("--config and --func must be provided together")

    seq_len, head_dim, head_num, batch, causal = parse_input_config(args.config)
    func_config = select_func_config(args.func)
    validate_case(func_config, seq_len, head_dim)

    label = shape_label(seq_len, head_dim, head_num, batch, causal)
    print(f"RUN {label} func={args.func}", flush=True)

    with torch.no_grad():
        q, k, v = make_inputs(seq_len, head_dim, head_num, batch, causal)
        out, elapsed_ms = run_custom(args.func, q, k, v, causal)

    throughput = batch * head_num * seq_len * head_dim / (elapsed_ms * 1.0e-3)
    print(
        f"DONE {label} func={args.func} elapsed_ms={elapsed_ms:.6f} "
        f"throughput={throughput / 1.0e9:.3f}G output_shape={tuple(out.shape)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
