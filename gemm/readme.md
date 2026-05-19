# GEMM CuTe

This directory builds and runs `test_gemm_cute`, a verification and benchmark
driver for the generated `gemm_tn` CuTe kernels in `gemm_cute.hpp`.

## Build

Run commands from the repository root unless noted otherwise.

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
