#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Built on NVIDIA CUTLASS / CuTe (BSD-3-Clause). See gemm/readme.md
# "Acknowledgements" and gemm/THIRD_PARTY_LICENSES.md for third-party notices.
"""Profile the gemm_tn best config per shape with Nsight Compute (ncu) and
report each kernel's SM (compute) throughput.

This mirrors flashattention/v3_mMnNkN/test_compute_rate.py, adapted to the
standalone `test_gemm_cute` C++ driver. The driver's `--profile` mode launches
one config on one shape with a single kernel launch fenced inside a
cudaProfilerStart/Stop region (after warmup launches); `--profile-from-start
off -c 1` then captures exactly that launch.

Typical flow:

  # 1. Produce BEST.txt (best config per shape) next to the binary.
  ./build/test_gemm_cute --benchmark

  # 2. Profile every best config from BEST.txt.
  python test_compute_rate.py --binary ./build/test_gemm_cute

  # Or profile a single config+shape directly (no BEST.txt needed):
  python test_compute_rate.py --binary ./build/test_gemm_cute \
      --func bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw4 --shape 4096 4096 4096

The SM compute rate read back is
`sm__throughput.avg.pct_of_peak_sustained_elapsed`; DRAM throughput
(`dram__throughput.avg.pct_of_peak_sustained_elapsed`) is reported alongside.
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path

SM_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

BEST_RE = re.compile(r"^BEST:\s*(\d+),(\d+),(\d+)\s+(\S+)")


def parse_best_file(path):
    """Return [(M, N, K, config_name), ...] parsed from a BEST.txt file."""
    entries = []
    with open(path) as f:
        for line in f:
            m = BEST_RE.match(line.strip())
            if m:
                M, N, K, cfg = m.groups()
                entries.append((int(M), int(N), int(K), cfg))
    return entries


def run_ncu(binary, config, shape, warmup, out_dir):
    """Profile one (config, shape) launch and return (sm_pct, dram_pct)."""
    M, N, K = shape
    tag = f"gemm__{config}__{M}x{N}x{K}"
    out_base = str(out_dir / tag)
    profile_cmd = [
        "ncu",
        "--profile-from-start", "off",
        "-c", "1",
        "--metrics", f"{SM_METRIC},{DRAM_METRIC}",
        "-o", out_base,
        "-f",
        binary, "--profile", config, str(M), str(N), str(K), str(warmup),
    ]
    proc = subprocess.run(profile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ncu failed for {tag} (rc={proc.returncode})")

    report = f"{out_base}.ncu-rep"
    csv_proc = subprocess.run(
        ["ncu", "-i", report, "--csv", "--page", "raw"],
        capture_output=True, text=True, check=True,
    )
    return extract_metrics(csv_proc.stdout)


def extract_metrics(csv_text):
    """Pull the SM and DRAM throughput percentages out of `ncu --csv` output."""
    def to_float(cell):
        if cell in (None, ""):
            return None
        try:
            return float(str(cell).replace(",", ""))
        except ValueError:
            return None

    reader = csv.DictReader(io.StringIO(csv_text))
    sm_val = dram_val = None
    for row in reader:
        sm = to_float(row.get(SM_METRIC))
        dram = to_float(row.get(DRAM_METRIC))
        if sm is not None:
            sm_val = sm
        if dram is not None:
            dram_val = dram
    return sm_val, dram_val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="./build/test_gemm_cute",
                    help="path to test_gemm_cute (default: ./build/test_gemm_cute)")
    ap.add_argument("--best-file", default=None,
                    help="BEST.txt path (default: BEST.txt next to the binary)")
    ap.add_argument("--func", default=None,
                    help="profile a single config instead of reading BEST.txt")
    ap.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                    help="shape for --func mode")
    ap.add_argument("--only", default=None,
                    help="comma-separated MxNxK filter, e.g. 2048x2048x2048,4096x4096x4096")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup launches before the profiled launch (default: 3)")
    ap.add_argument("--out-dir", default=None,
                    help="directory for .ncu-rep files (default: <repo>/tmp/ncu)")
    args = ap.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        ap.error(f"binary not found: {binary}")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "tmp" / "ncu"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.func:
        if not args.shape:
            ap.error("--func requires --shape M N K")
        entries = [(args.shape[0], args.shape[1], args.shape[2], args.func)]
    else:
        best_file = args.best_file or os.path.join(os.path.dirname(binary), "BEST.txt")
        if not os.path.exists(best_file):
            ap.error(f"BEST.txt not found: {best_file}\n"
                     f"Run `{binary} --benchmark` first, or pass --best-file.")
        entries = parse_best_file(best_file)

    if args.only:
        keep = {tuple(int(x) for x in s.lower().split("x")) for s in args.only.split(",")}
        entries = [e for e in entries if (e[0], e[1], e[2]) in keep]

    if not entries:
        ap.error("no shapes to profile")

    print(f"# binary: {binary}")
    print(f"{'shape':<22} {'config':<46} {'SM %':>8} {'DRAM %':>8}")
    for M, N, K, cfg in entries:
        shape_label = f"{M}x{N}x{K}"
        try:
            sm, dram = run_ncu(binary, cfg, (M, N, K), args.warmup, out_dir)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"{shape_label:<22} {cfg:<46} {'ERR':>8} {'ERR':>8}  ({exc})")
            continue
        sm_s = f"{sm:.2f}" if sm is not None else "n/a"
        dram_s = f"{dram:.2f}" if dram is not None else "n/a"
        print(f"{shape_label:<22} {cfg:<46} {sm_s:>8} {dram_s:>8}")


if __name__ == "__main__":
    main()
