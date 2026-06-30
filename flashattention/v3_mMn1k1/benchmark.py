#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func

OUTPUT = Path(__file__).resolve().parent / "benchmark.txt"
test_shapes = [
    (2048, 64, 16, 1),
    (2048, 64, 32, 8),
    (4096, 128, 32, 1),
    (4096, 128, 64, 2),
    (4096, 128, 64, 8),
    (8192, 128, 32, 1),
    (8192, 128, 80, 4),
    (16384, 128, 32, 1),
]


sys.path.insert(0, str(Path(__file__).resolve().parent / "pybind" / "build"))
import flashatten_v3_sm120


OFFICIAL_NAME = "official_flashatten2"


def parse_bool(value):
    if value in ("1", "true", "True", "yes", "y"):
        return True
    if value in ("0", "false", "False", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def select_configs(config_name):
    configs = flashatten_v3_sm120.configs()
    if not config_name:
        return configs
    selected = [cfg for cfg in configs if cfg[0] == config_name]
    if not selected:
        names = ", ".join(cfg[0] for cfg in configs)
        raise SystemExit(f"unknown config '{config_name}'. available: {names}")
    return selected


def select_causal_modes(mode, all_shapes):
    if mode is None:
        return (False, True) if all_shapes else (False,)
    if mode == "both":
        return (False, True)
    return (parse_bool(mode),)


def config_fits(config, shape):
    _, config_head_dim, block_m, block_n, _, _, _ = config
    seq_len, head_dim, _, _ = shape
    return head_dim == config_head_dim and seq_len % block_m == 0 and seq_len % block_n == 0


def make_inputs(shape, causal):
    seq_len, head_dim, head_num, batch = shape

    torch.manual_seed(20260617 + seq_len + head_dim + head_num + batch + int(causal))
    q = (torch.randn(batch, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16) * 0.5).contiguous()
    k = (torch.randn_like(q) * 0.5).contiguous()
    v = (torch.randn_like(q) * 0.5).contiguous()
    return q, k, v


def time_call(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def throughput(shape, avg_ms):
    seq_len, head_dim, head_num, batch = shape
    return batch * head_num * seq_len * head_dim / (avg_ms * 1.0e-3)


def format_throughput(value):
    for suffix in ("", "K", "M", "G", "T"):
        if abs(value) < 1000.0:
            return f"{value:.3f}{suffix}"
        value /= 1000.0
    return f"{value:.3f}P"


def benchmark_config(config, shape, causal, q, k, v, warmup, iterations):
    name, _, _, _, _, _, _ = config
    avg_ms = time_call(lambda: flashatten_v3_sm120.fwd_config(name, q, k, v, causal), warmup, iterations)
    return name, throughput(shape, avg_ms), avg_ms


def benchmark_official(shape, causal, q, k, v, warmup, iterations):
    _, head_dim, _, _ = shape
    scale = 1.0 / math.sqrt(head_dim)
    avg_ms = time_call(
        lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=causal),
        warmup,
        iterations,
    )
    return OFFICIAL_NAME, throughput(shape, avg_ms), avg_ms


def shape_label(shape, causal):
    seq_len, head_dim, head_num, batch = shape
    return f"b{batch}_hn{head_num}_seq{seq_len}_hd{head_dim}_casual{int(causal)}"


def benchmark_shape(configs, shape, causal, warmup, iterations):
    q, k, v = make_inputs(shape, causal)
    basic_results = []
    with torch.no_grad():
        for config in configs:
            if config_fits(config, shape):
                basic_results.append(benchmark_config(config, shape, causal, q, k, v, warmup, iterations))
        official_result = benchmark_official(shape, causal, q, k, v, warmup, iterations)

    label = shape_label(shape, causal)
    for name, thru, avg_ms in sorted(basic_results, key=lambda item: item[1], reverse=True):
        print(f"{label} {name} {format_throughput(thru)} {avg_ms:.6f}", flush=True)

    del q, k, v
    torch.cuda.empty_cache()
    if not basic_results:
        return None
    best_basic = max(basic_results, key=lambda item: item[1])
    ratio = best_basic[1] / official_result[1] * 100.0
    return (
        f"{label} {best_basic[0]}: {format_throughput(best_basic[1])} {best_basic[2]:.6f} "
        f"{official_result[0]}: {format_throughput(official_result[1])} {official_result[2]:.6f} "
        f"custom/official:{ratio:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Benchmark one generated config by exact name")
    parser.add_argument("--causal", help="both, true, or false")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--case", nargs=4, type=int, metavar=("SEQ_LEN", "HEAD_DIM", "HEAD_NUM", "BATCH"))
    parser.add_argument("--all", action="store_true", help="Benchmark enabled shapes from verify.py")
    args = parser.parse_args()

    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    shapes = test_shapes if args.all or not args.case else [tuple(args.case)]
    configs = select_configs(args.config)
    causal_modes = select_causal_modes(args.causal, args.all)

    summary_lines = []
    for shape in shapes:
        for causal in causal_modes:
            line = benchmark_shape(configs, shape, causal, args.warmup, args.iterations)
            if line:
                summary_lines.append(line)

    OUTPUT.write_text("\n".join(summary_lines) + ("\n" if summary_lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
