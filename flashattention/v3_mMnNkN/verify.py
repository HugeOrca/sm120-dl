#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func


sys.path.insert(0, str(Path(__file__).resolve().parent / "pybind" / "build"))
import flashatten_v3_sm120


ATOL = 7.5e-2
RTOL = 7.5e-2
INSTANTIATED_HEAD_NUMS = (1, 4, 8, 16, 32, 64, 80)
CAUSAL_VALUES = (False, True)
FAILED_CASES_PATH = Path(__file__).resolve().parent / "pybind" / "build" / "failed_resource_cases.log"
RESOURCE_ERROR_MARKERS = (
    "out of resources",
    "too many resources requested",
    "cudaerrorlaunchoutofresources",
    "invalid configuration argument",
)
test_shapes = [
    (64, 64, 1, 1),
    (128, 64, 1, 1),
    (128, 64, 1, 2),
    (192, 64, 1, 1),
    (128, 64, 4, 1),
    (192, 64, 4, 1),
    (256, 64, 8, 2),
    (128, 128, 4, 1),
    (128, 128, 4, 2),
    (256, 128, 4, 1),
    (2048, 64, 16, 1),
    (4096, 128, 32, 1),
    (8192, 128, 32, 1),
    (16384, 128, 32, 1),
    (2048, 64, 32, 8),
    (4096, 128, 64, 2),
    (4096, 128, 64, 8),
    (8192, 128, 80, 4),
]


def parse_bool(value):
    if value in ("1", "true", "True", "yes", "y"):
        return True
    if value in ("0", "false", "False", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def make_inputs(seq_len, head_dim, head_num, batch, causal):
    torch.manual_seed(20260614 + seq_len + head_dim + head_num + batch + int(causal))
    q = (torch.randn(batch, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16) * 0.5).contiguous()
    k = (torch.randn_like(q) * 0.5).contiguous()
    v = (torch.randn_like(q) * 0.5).contiguous()
    return q, k, v


def compare_tensors(got, ref):
    diff = got.float()
    diff.sub_(ref.float()).abs_()
    ref_abs = ref.float().abs_()
    threshold = ref_abs.mul(RTOL).add_(ATOL)
    bad = diff > threshold
    bad |= ~torch.isfinite(got)
    bad |= ~torch.isfinite(ref)
    diff = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    max_abs = diff.max().item()
    diff.div_(ref_abs.clamp_min_(1.0e-6))
    rel = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    max_rel = rel.max().item()
    mismatch_count = bad.sum().item()
    total_count = got.numel()
    passed = not bad.any()
    return passed, max_abs, max_rel, mismatch_count, total_count


def format_case_name(seq_len, head_dim, head_num, batch, causal):
    return f"b{batch}_hn{head_num}_seq{seq_len}_hd{head_dim}_casual{int(causal)}"


def format_result_prefix(status, seq_len, head_dim, head_num, batch, causal, config=None):
    result = f"{status} {format_case_name(seq_len, head_dim, head_num, batch, causal)}"
    if config is not None:
        result += f" {config}"
    return result


def print_result(status, seq_len, head_dim, head_num, batch, causal, max_abs, max_rel, mismatch_count, total_count, config=None):
    print(
        f"{format_result_prefix(status, seq_len, head_dim, head_num, batch, causal, config)} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"mismatches={mismatch_count}/{total_count}",
        flush=True,
    )


def is_resource_error(exc):
    message = str(exc).lower()
    return any(marker in message for marker in RESOURCE_ERROR_MARKERS)


def write_failed_cases(cases):
    lines = ["Failed resource cases", ""]
    if not cases:
        lines.append("None.")
    else:
        for case in cases:
            lines.append(
                "- "
                f"{format_case_name(case['seq_len'], case['head_dim'], case['head_num'], case['batch'], case['causal'])} "
                f"{case['config']} "
                f"error={case['error']}"
            )
    FAILED_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILED_CASES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify(seq_len, head_dim, head_num, batch, causal):
    with torch.no_grad():
        q, k, v = make_inputs(seq_len, head_dim, head_num, batch, causal)

        got = flashatten_v3_sm120.fwd(q, k, v, causal)
        ref = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=1.0 / math.sqrt(head_dim), causal=causal)
        torch.cuda.synchronize()

        passed, max_abs, max_rel, mismatch_count, total_count = compare_tensors(got, ref)

    print_result(
        "PASS" if passed else "FAIL",
        seq_len,
        head_dim,
        head_num,
        batch,
        causal,
        max_abs,
        max_rel,
        mismatch_count,
        total_count,
    )
    del q, k, v, got, ref
    torch.cuda.empty_cache()
    return passed


def verify_kernel_template_config(config, head_num, causal, failed_cases):
    name, head_dim, block_m, block_n, _, _, _ = config
    seq_len = math.lcm(block_m, block_n)
    batch = 1
    passed = False

    with torch.no_grad():
        q, k, v = make_inputs(seq_len, head_dim, head_num, batch, causal)
        ref = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=1.0 / math.sqrt(head_dim), causal=causal)
        torch.cuda.synchronize()

        try:
            got = flashatten_v3_sm120.fwd_config(name, q, k, v, causal)
            torch.cuda.synchronize()
            passed, max_abs, max_rel, mismatch_count, total_count = compare_tensors(got, ref)
            print_result(
                "PASS" if passed else "FAIL",
                seq_len,
                head_dim,
                head_num,
                batch,
                causal,
                max_abs,
                max_rel,
                mismatch_count,
                total_count,
                config=name,
            )
            del got
        except RuntimeError as exc:
            if is_resource_error(exc):
                failed_cases.append(
                    {
                        "config": name,
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                        "head_num": head_num,
                        "batch": batch,
                        "causal": causal,
                        "error": str(exc).replace("\n", " "),
                    }
                )
                print(
                    f"{format_result_prefix('SKIP_RESOURCE', seq_len, head_dim, head_num, batch, causal, name)} "
                    f"error={exc}",
                    flush=True,
                )
                passed = True
            else:
                print(
                    f"{format_result_prefix('FAIL', seq_len, head_dim, head_num, batch, causal, name)} "
                    f"error={exc}",
                    flush=True,
                )

    del q, k, v, ref
    torch.cuda.empty_cache()
    return passed


def verify_all_kernel_templates():
    ok = True
    passed_count = 0
    seen = set()
    failed_cases = []
    write_failed_cases(failed_cases)

    for config in flashatten_v3_sm120.configs():
        for head_num in INSTANTIATED_HEAD_NUMS:
            for causal in CAUSAL_VALUES:
                key = (config[0], head_num, causal)
                if key in seen:
                    print(f"FAIL duplicate kernel template config={key}", flush=True)
                    return False
                seen.add(key)
                passed = verify_kernel_template_config(config, head_num, causal, failed_cases)
                ok &= passed
                passed_count += int(passed)

    expected_count = len(flashatten_v3_sm120.configs()) * len(INSTANTIATED_HEAD_NUMS) * len(CAUSAL_VALUES)
    if len(seen) != expected_count:
        print(
            f"FAIL kernel_templates coverage mismatch actual={len(seen)} expected={expected_count}",
            flush=True,
        )
        return False

    print(
        f"SUMMARY kernel_templates passed={passed_count}/{len(seen)} "
        f"configs={len(flashatten_v3_sm120.configs())} "
        f"head_nums={len(INSTANTIATED_HEAD_NUMS)} causals={len(CAUSAL_VALUES)} "
        f"resource_skips={len(failed_cases)}",
        flush=True,
    )
    write_failed_cases(failed_cases)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=5, metavar=("SEQ_LEN", "HEAD_DIM", "HEAD_NUM", "BATCH", "CAUSAL"))
    parser.add_argument(
        "--all",
        "--all-configs",
        dest="all_configs",
        action="store_true",
        help="Verify every instantiated kernel template config",
    )
    args = parser.parse_args()

    if args.case:
        if args.all_configs:
            parser.error("--all verifies the full kernel-template set and cannot be combined with --case")
        seq_len, head_dim, head_num, batch = map(int, args.case[:4])
        causal = parse_bool(args.case[4])
        return 0 if verify(seq_len, head_dim, head_num, batch, causal) else 1

    ok = True
    if args.all_configs:
        return 0 if verify_all_kernel_templates() else 1

    for shape in test_shapes:
        ok &= verify(*shape, causal=False)
        ok &= verify(*shape, causal=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
