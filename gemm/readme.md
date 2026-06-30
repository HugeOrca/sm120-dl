# GEMM CuTe

This directory builds and runs `test_gemm_cute`, a verification and benchmark
driver for the generated `gemm_tn` CuTe kernels in `gemm_cute.hpp`.

## Build

Build commands start from the repository root (each snippet runs `cd gemm`
first); the run/benchmark commands below are issued from `gemm/`. In the
snippets, `-j16` runs 16 parallel compile jobs — set it to your CPU core count.

Default full build, instantiating all generated `gemm_tn` configs:
```bash
cd gemm
cmake -S . -B build
cd build
make -j16 test_gemm_cute
```

Build only one generated config:

```bash
cd gemm
cmake -S . -B build_one -DGEMM_TN_ONLY_CONFIG=bf16_bm64_bn128_bk64_s3_cwg2_wm2_wn4_bsw4
cmake --build build_one --target test_gemm_cute -j16
```

Useful CMake cache parameters:

```bash
-DGEMM_TN_MAX_SMEM_KB=<KiB>
-DGEMM_TN_ONLY_CONFIG=<config-name>
```

The generated files are written under the selected build directory, for example
`gemm/build/generated` or `gemm/build_one/generated`.

## Run

Run commands from `gemm/` after building.

Verify every generated config on all built-in shapes:

```bash
./build/test_gemm_cute --verify
```

Benchmark every generated config on all built-in shapes:

```bash
./build/test_gemm_cute --benchmark
```

Verify and benchmark one config on the fixed `M=N=K=4096` shape:

```bash
./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4
```

Run one config in verify-only or benchmark-only mode:

```bash
./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4 --verify
./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4 --benchmark
```

Verify and benchmark every generated config on a custom shape:

```bash
./build/test_gemm_cute <M> <N> <K>
```

`<config>` accepts generated config names with or without the `gemm_tn_`
prefix. A trailing `*` is accepted as a prefix pattern if it matches exactly one
config. Runtime execution skips configs whose `BM`, `BN`, or `BK` do not divide
the selected `M`, `N`, or `K`.

## Compute Throughput (ncu)

`test_gemm_cute --profile <config> <M> <N> <K> [warmup]` runs `warmup` launches,
then issues exactly one kernel launch inside a `cudaProfilerStart` /
`cudaProfilerStop` region. This isolates a single launch so Nsight Compute can
profile only that kernel (the same pattern as
`flashattention/v3_mMnNkN/test_compute_rate.py`).

`test_compute_rate.py` drives ncu over the best config per shape. First produce
`BEST.txt` (best config per shape) with a benchmark run, then profile each entry:

```bash
cd gemm
./build/test_gemm_cute --benchmark           # writes ./build/BEST.txt
python3 test_compute_rate.py --binary ./build/test_gemm_cute
```

It reports `sm__throughput.avg.pct_of_peak_sustained_elapsed` (SM compute rate)
and `dram__throughput.avg.pct_of_peak_sustained_elapsed` for each kernel. Profile
a single config + shape directly (no `BEST.txt` needed):

```bash
python3 test_compute_rate.py --binary ./build/test_gemm_cute \
    --func bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw4 --shape 4096 4096 4096
```

`test_compute_rate.py` flags:

- `--binary`: path to `test_gemm_cute` (default `./build/test_gemm_cute`).
- `--best-file`: `BEST.txt` to read (default: next to the binary).
- `--func` + `--shape M N K`: profile one config on one shape instead of `BEST.txt`.
- `--only`: comma-separated `MxNxK` filter, e.g. `2048x2048x2048,4096x4096x4096`.
- `--warmup`: warmup launches before the profiled launch (default `3`).
- `--out-dir`: directory for `.ncu-rep` files (default `<repo>/tmp/ncu`).

Run after locking GPU frequency for stable numbers.

## Throughput vs cuBLAS (best config per shape)

End-to-end throughput from `./build/test_gemm_cute --benchmark` (bf16, 5 warmup +
25 timed iterations), best `gemm_tn` config per shape against `cublasGemmEx`
(`CUBLAS_GEMM_DEFAULT_TENSOR_OP`). GPU clocks were locked (`nvidia-smi -lgc`), so
the numbers are stable run-to-run. `gemm_tn/cuBLAS` > 100 % means `gemm_tn` is
faster.

| Shape (M×N×K)       | gemm_tn ms | gemm_tn TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s | gemm_tn/cuBLAS |
| ---                 | ---:       | ---:            | ---:      | ---:           | ---:           |
| `128×4096×14336`    |   0.1882   |  79.87          |   0.2288  |  65.71         | **121.6 %**    |
| `128×28672×4096`    |   0.3481   |  86.36          |   0.3744  |  80.29         | 107.6 %        |
| `512×512×14336`     |   0.0921   |  81.57          |   0.0869  |  86.49         | 94.3 %         |
| `1024×1024×1024`    |   0.0343   |  62.68          |   0.0261  |  82.29         | 76.2 %         |
| `1024×1024×14336`   |   0.3405   |  88.29          |   0.3410  |  88.16         | 100.2 %        |
| `2048×2048×2048`    |   0.1919   |  89.51          |   0.1988  |  86.40         | 103.6 %        |
| `4096×4096×4096`    |   1.5009   |  91.57          |   1.5580  |  88.22         | 103.8 %        |
| `4096×4096×14336`   |   5.1935   |  92.62          |   5.3229  |  90.37         | 102.5 %        |
| `4096×28672×4096`   |  10.2467   |  93.89          |  10.5905  |  90.84         | 103.4 %        |
| `8192×8192×8192`    |  12.1053   |  90.83          |  12.3796  |  88.82         | 102.3 %        |
| `16384×16384×16384` |  91.5637   |  96.07          |  96.6524  |  91.01         | 105.6 %        |
| `40960×40960×4096`  | 139.4008   |  98.59          | 143.5932  |  95.71         | 103.0 %        |

Best config per shape:

```text
128x4096x14336      bf16_bm64_bn128_bk64_s3_cwg2_wm2_wn4_bsw4
128x28672x4096      bf16_bm128_bn64_bk64_s3_cwg2_wm4_wn2_bsw8
512x512x14336       bf16_bm64_bn64_bk64_s3_cwg2_wm4_wn2_bsw8
1024x1024x1024      bf16_bm64_bn64_bk64_s2_cwg2_wm4_wn2_bsw8
1024x1024x14336     bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw8
2048x2048x2048      bf16_bm64_bn64_bk64_s3_cwg2_wm4_wn2_bsw8
4096x4096x4096      bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw8
4096x4096x14336     bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw8
4096x28672x4096     bf16_bm128_bn64_bk64_s2_cwg2_wm4_wn2_bsw4
8192x8192x8192      bf16_bm128_bn64_bk64_s3_cwg2_wm2_wn4_bsw8
16384x16384x16384   bf16_bm128_bn128_bk64_s2_cwg2_wm4_wn2_bsw4
40960x40960x4096    bf16_bm128_bn128_bk64_s2_cwg2_wm2_wn4_bsw8
```

`gemm_tn` matches or beats cuBLAS on every shape with enough work to fill the GPU
(≥ `2048³`), peaking at **+21.6 %** on the thin `128×4096×14336` case and holding
**+3–5 %** on the large square/long-K shapes. It only trails on tiny,
launch-bound problems (`1024³`, `512×512×14336`) where cuBLAS's lighter-weight
kernels win.

## Acknowledgements

The `gemm_tn` kernels are original project code built on NVIDIA CUTLASS / CuTe,
which is licensed under the BSD 3-Clause License:

- NVIDIA CUTLASS / CuTe — Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
  <https://github.com/NVIDIA/cutlass>

Every source file in this directory carries a short SPDX header. The full
upstream copyright notice and BSD 3-Clause license text are reproduced in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## License

This project's own code is licensed under Apache-2.0 (see the repository
`LICENSE` file). It builds on third-party components under the BSD 3-Clause
License; their copyright notices and full license texts are in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
