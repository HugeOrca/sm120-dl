# SM120 Deep Learning Kernels

This project provides high-performance deep learning kernel implementations and experimental code for NVIDIA SM120 GPUs.

## Background

CUDA, cuBLAS, and CUTLASS support for SM120 is still evolving, and coverage for production-ready high-performance kernels is not yet complete in every deep learning workload. This makes it difficult for developers to directly obtain mature operator implementations for SM120.

This project aims to fill those gaps by implementing and optimizing common deep learning kernels around the SM120 architecture. The goal is to provide verifiable and benchmarkable building blocks for real model training and inference workloads.

Current focus areas include:

- GEMM kernel implementations and configuration generation for SM120.
- Clear example code based on the CuTe/CUTLASS programming model.
- Kernel correctness validation, performance benchmarking, and tuning experiments.

## Goals

This project has two main goals:

1. **Provide high-performance deep learning kernels for SM120**

   The project targets SM120 scenarios that are not yet fully covered or optimized by CUDA, cuBLAS, and CUTLASS. It provides high-performance kernels that can be studied, extended, and adapted directly. The current implementation focus is GEMM, which is one of the most important compute paths in deep learning.

2. **Serve as a learning resource for SM120 and CUTLASS/CuTe**

   The code is kept simple, explicit, and readable, avoiding excessive abstraction around the core logic. Developers can use these implementations to understand SM120 architectural features, CuTe/CUTLASS kernel development, data movement, tiling, shared memory usage, warp-group collaboration, and performance tuning techniques.

## Project Structure

```text
.
├── gemm/             # SM120 GEMM kernels, configuration generation, validation, and benchmarks
├── utils/            # Shared utility code
└── 3rd/cutlass/      # CUTLASS/CuTe dependency
```

## Quick Start

The GEMM module includes its own build, validation, and benchmark flow. See the module README for detailed commands:

- [gemm/readme.md](gemm/readme.md)

Common build commands:

```bash
cd gemm
cmake -S . -B build
cmake --build build --target test_gemm_cute -j16
```

Validation and benchmarking:

```bash
./build/test_gemm_cute --verify
./build/test_gemm_cute --benchmark
```

## Development Principles

- Implement and tune kernels around SM120 architectural characteristics.
- Prioritize kernels with verifiable correctness, benchmarkable performance, and reproducible experimental results.
- Keep the code straightforward so it remains easy to read, modify, and extend.
- Treat the project as both an optimization effort and a learning resource, with clear structure and readable implementations.

## Current Status

This project is under active development. The current focus is improving the SM120 GEMM kernels and gradually documenting more optimization techniques and experimental results.
