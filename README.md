# SM120 Deep Learning Kernels

This project provides high-performance deep learning kernel implementations and experimental code for NVIDIA SM120 GPUs.

## Background

CUDA, cuBLAS, and CUTLASS support for SM120 is still evolving, and coverage for production-ready high-performance kernels is not yet complete in every deep learning workload. This makes it difficult for developers to directly obtain mature operator implementations for SM120.

This project aims to fill those gaps by implementing and optimizing common deep learning kernels around the SM120 architecture. The goal is to provide verifiable and benchmarkable building blocks for real model training and inference workloads.

Current focus areas include:

- GEMM kernel implementations and configuration generation for SM120.
- FlashAttention forward kernels with two MMA layouts and Python bindings.
- Clear example code based on the CuTe/CUTLASS programming model.
- Kernel correctness validation, performance benchmarking, and tuning experiments.

## Goals

This project has two main goals:

1. **Provide high-performance deep learning kernels for SM120**

   The project targets SM120 scenarios that are not yet fully covered or optimized by CUDA, cuBLAS, and CUTLASS. It provides high-performance kernels that can be studied, extended, and adapted directly. Current implementations include BF16 GEMM and FP16 FlashAttention forward attention.

2. **Serve as a learning resource for SM120 and CUTLASS/CuTe**

   The code is kept simple, explicit, and readable, avoiding excessive abstraction around the core logic. Developers can use these implementations to understand SM120 architectural features, CuTe/CUTLASS kernel development, data movement, tiling, shared memory usage, warp-group collaboration, and performance tuning techniques.

## Project Structure

```text
.
├── gemm/                       # BF16 GEMM kernels, config generation, validation, and benchmarks
├── flashattention/
│   ├── v3_mMn1k1/              # Forward attention: QK/PV MMA layout (M, 1, 1)
│   └── v3_mMnNkN/              # Forward attention: QK (M, N, 1), PV (M, 1, N)
├── utils/                      # Shared utility code
├── lock_gpu_freq.sh            # GPU clock control for benchmarks
└── 3rd/cutlass/                # CUTLASS/CuTe dependency (Git submodule)
```

## Quick Start

Run the commands below from the repository root. All modules require an NVIDIA
SM120 GPU, a CUDA Toolkit/compiler with `sm_120a` support, CMake `>= 3.24`,
and C++20/CUDA20 support. Initialize the CUTLASS dependency first:

```bash
git submodule update --init --recursive
export REPO_ROOT="$(git rev-parse --show-toplevel)"
```

Each module documents its build, validation, benchmarking, and profiling flow:

- [GEMM](gemm/README.md)
- [FlashAttention: v3_mMn1k1](flashattention/v3_mMn1k1/README.md)
- [FlashAttention: v3_mMnNkN](flashattention/v3_mMnNkN/README.md)

Build, verify, and benchmark GEMM (requires `python3` for config generation):

```bash
cmake -S "$REPO_ROOT/gemm" -B "$REPO_ROOT/gemm/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$REPO_ROOT/gemm/build" --target test_gemm_cute -j16
"$REPO_ROOT/gemm/build/test_gemm_cute" --verify
"$REPO_ROOT/gemm/build/test_gemm_cute" --benchmark
```

Build, verify, and benchmark a FlashAttention variant. The current build expects
a conda environment with CUDA 13 PyTorch, `pybind11`, Ninja, and
`nvidia/cu13/lib/libcudart.so.13`; validation and benchmarking also require
official `flash-attn`. Replace `torch132` with your environment name:

```bash
conda activate torch132
export TORCH132_ENV="$CONDA_PREFIX"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$TORCH132_ENV/lib:${LD_LIBRARY_PATH:-}"

# Use v3_mMn1k1 here to build the other implementation.
export FLASH_VARIANT=v3_mMnNkN
export PYBIND_BUILD="$REPO_ROOT/flashattention/$FLASH_VARIANT/pybind/build"
cmake -S "$REPO_ROOT/flashattention/$FLASH_VARIANT/pybind" -B "$PYBIND_BUILD" \
  -G Ninja \
  -DTORCH132_ENV="$TORCH132_ENV" \
  -DCMAKE_MAKE_PROGRAM="$TORCH132_ENV/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$PYBIND_BUILD" -j8

cd "$REPO_ROOT/flashattention/$FLASH_VARIANT"
python verify.py --all
python benchmark.py --all
```

Both variants build a module named `flashatten_v3_sm120`. Run each variant in a
separate Python process using its own `pybind/build` directory.

## Development Principles

- Implement and tune kernels around SM120 architectural characteristics.
- Prioritize kernels with verifiable correctness, benchmarkable performance, and reproducible experimental results.
- Keep the code straightforward so it remains easy to read, modify, and extend.
- Treat the project as both an optimization effort and a learning resource, with clear structure and readable implementations.

## Current Status

This project is under active development. BF16 GEMM and both FP16 FlashAttention
forward variants include config generation, correctness validation, benchmarks,
and Nsight Compute profiling tools. The FlashAttention bindings support matching
Q/K/V shapes and require sequence lengths divisible by the selected kernel's
block sizes; see the module READMEs for supported head dimensions and head counts.
