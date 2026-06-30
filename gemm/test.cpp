// SPDX-License-Identifier: Apache-2.0
// Built on NVIDIA CUTLASS / CuTe (BSD-3-Clause). See gemm/readme.md
// "Acknowledgements" and gemm/THIRD_PARTY_LICENSES.md for third-party notices.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unistd.h>

#include "gemm_config_registry.hpp"

namespace {

constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 25;
constexpr float kAbsTolerance = 1.0e-2f;
constexpr float kRelTolerance = 5.0e-2f;

struct GemmShape {
    int M;
    int N;
    int K;
};

constexpr GemmShape kTestShapes[] = {
    // {1024, 1024, 1024},
    // {2048, 2048, 2048},
    // {4096, 4096, 4096},
    // {8192, 8192, 8192},
    // {3456, 4096, 4096},
    // {4096, 7168, 2048},
    // {4096, 4096, 7168},
    // {4096, 7168, 16384},
    // {4096, 24576, 1536},
    // {4096, 32768, 512},
    {128, 4096, 14336},
    {128, 28672, 4096},
    {512, 512, 14336},
    {1024, 1024, 1024},
    {1024, 1024, 14336},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {4096, 4096, 14336},
    {4096, 28672, 4096},
    {8192, 8192, 8192},
    {16384, 16384, 16384},
    {40960, 40960, 4096},
};

const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS status";
    }
}

bool check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s:%d CUDA error at %s: %s\n", file, line, expr, cudaGetErrorString(status));
    return false;
}

bool check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s:%d cuBLAS error at %s: %s\n", file, line, expr, cublas_status_to_string(status));
    return false;
}

#define CHECK_CUDA(expr) \
    do { \
        if (!check_cuda((expr), #expr, __FILE__, __LINE__)) { \
            return EXIT_FAILURE; \
        } \
    } while (0)

#define CHECK_CUBLAS(expr) \
    do { \
        if (!check_cublas((expr), #expr, __FILE__, __LINE__)) { \
            return EXIT_FAILURE; \
        } \
    } while (0)

template <class T>
void fill_random(thrust::host_vector<T>& tensor, float min_value, float max_value) {
    static std::mt19937 rng(20260509);
    std::uniform_real_distribution<float> dist(min_value, max_value);

    for (auto& value : tensor) {
        value = static_cast<T>(dist(rng));
    }
}

template <class T>
constexpr cudaDataType_t cublas_data_type() {
    if constexpr (std::is_same_v<T, bf16>) {
        return CUDA_R_16BF;
    } else if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return CUDA_R_16F;
    } else {
        static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, cutlass::half_t>,
                      "unsupported cuBLAS data type");
    }
}

template <class T>
cublasStatus_t launch_cublas_gemm(
    cublasHandle_t handle,
    int M,
    int N,
    int K,
    const T* A,
    const T* B,
    T* C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    return cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B,
        cublas_data_type<T>(),
        K,
        A,
        cublas_data_type<T>(),
        K,
        &beta,
        C,
        cublas_data_type<T>(),
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template <class T>
bool verify_result(
    const thrust::host_vector<T>& actual,
    const thrust::host_vector<T>& reference,
    int cols,
    float abs_tol,
    float rel_tol,
    bool print_detail) {
    if (actual.size() != reference.size()) {
        std::fprintf(stderr, "size mismatch: actual=%zu reference=%zu\n", actual.size(), reference.size());
        return false;
    }

    size_t mismatches = 0;
    size_t first_mismatch = actual.size();
    size_t max_idx = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        const float got = static_cast<float>(actual[i]);
        const float ref = static_cast<float>(reference[i]);
        const float abs_diff = std::abs(got - ref);
        const float rel_diff = abs_diff / std::max(std::abs(ref), 1.0e-20f);
        const float allowed = abs_tol + rel_tol * std::abs(ref);

        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            max_rel = rel_diff;
            max_idx = i;
        }
        if (abs_diff > allowed) {
            if (first_mismatch == actual.size()) {
                first_mismatch = i;
            }
            ++mismatches;
        }
    }

    if (print_detail) {
        std::printf(
            "verify detail: max_abs=%g max_rel=%g max_idx=%zu mismatches=%zu/%zu\n",
            max_abs,
            max_rel,
            max_idx,
            mismatches,
            actual.size());
    }

    if (mismatches != 0) {
        const size_t row = first_mismatch / static_cast<size_t>(cols);
        const size_t col = first_mismatch % static_cast<size_t>(cols);
        if (print_detail) {
            std::fprintf(
                stderr,
                "first mismatch at linear index=%zu row=%zu col=%zu got=%g ref=%g\n",
                first_mismatch,
                row,
                col,
                static_cast<float>(actual[first_mismatch]),
                static_cast<float>(reference[first_mismatch]));
        }
        return false;
    }
    return true;
}

template <class Launcher>
bool benchmark_launcher(
    cudaStream_t stream,
    int warmup_iterations,
    int benchmark_iterations,
    Launcher&& launch,
    float& avg_ms) {
    for (int i = 0; i < warmup_iterations; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize()", __FILE__, __LINE__)) {
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(&start)", __FILE__, __LINE__)) {
        return false;
    }
    if (!check_cuda(cudaEventCreate(&stop), "cudaEventCreate(&stop)", __FILE__, __LINE__)) {
        cudaEventDestroy(start);
        return false;
    }

    auto destroy_events = [&]() {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    };

    if (!check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start, stream)", __FILE__, __LINE__)) {
        destroy_events();
        return false;
    }
    for (int i = 0; i < benchmark_iterations; ++i) {
        if (!launch()) {
            destroy_events();
            return false;
        }
    }
    if (!check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop, stream)", __FILE__, __LINE__) ||
        !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)", __FILE__, __LINE__)) {
        destroy_events();
        return false;
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                    "cudaEventElapsedTime(&elapsed_ms, start, stop)",
                    __FILE__,
                    __LINE__)) {
        destroy_events();
        return false;
    }

    destroy_events();
    avg_ms = elapsed_ms / static_cast<float>(benchmark_iterations);
    return true;
}

double gemm_tflops(int M, int N, int K, float latency_ms) {
    if (latency_ms <= 0.0f) {
        return 0.0;
    }
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return flops / (static_cast<double>(latency_ms) * 1.0e9);
}

double throughput_percent(double gemm_tn_tflops, double cublas_tflops) {
    if (cublas_tflops <= 0.0) {
        return 0.0;
    }
    return gemm_tn_tflops / cublas_tflops * 100.0;
}

void print_shape_braced(const GemmShape& shape) {
    std::printf("{%d, %d, %d}", shape.M, shape.N, shape.K);
}

struct BenchmarkResult {
    bool ok = false;
    bool skipped = false;
    float latency_ms = 0.0f;
    double throughput = 0.0;
};

struct BenchmarkSummary {
    const GemmConfig* config = nullptr;
    BenchmarkResult result;
    size_t original_index = 0;
};

struct BenchmarkBest {
    GemmShape shape{};
    const GemmConfig* config = nullptr;
    BenchmarkResult result;
    float cublas_latency_ms = 0.0f;
    double cublas_throughput = 0.0;
};

struct VerifySummary {
    bool ok = true;
    size_t passed = 0;
    size_t failed = 0;
    size_t skipped = 0;
    size_t total = 0;
};

struct BenchmarkShapeResult {
    bool ok = true;
    BenchmarkBest best;
    size_t passed = 0;
    size_t failed = 0;
    size_t skipped = 0;
    size_t total = 0;
};

enum class FuncMode {
    VerifyAndBenchmark,
    VerifyOnly,
    BenchmarkOnly,
};

const char* func_mode_name(FuncMode mode) {
    switch (mode) {
        case FuncMode::VerifyAndBenchmark: return "verify+benchmark";
        case FuncMode::VerifyOnly: return "verify";
        case FuncMode::BenchmarkOnly: return "benchmark";
    }
    return "unknown";
}

template <class T>
struct GemmProblem {
    int M;
    int N;
    int K;
    thrust::host_vector<T> host_A;
    thrust::host_vector<T> host_B;
    thrust::host_vector<T> host_C;
    thrust::device_vector<T> device_A;
    thrust::device_vector<T> device_B;
    thrust::device_vector<T> device_C;

    GemmProblem(int m, int n, int k)
        : M(m),
          N(n),
          K(k),
          host_A(static_cast<size_t>(m) * k),
          host_B(static_cast<size_t>(n) * k),
          host_C(static_cast<size_t>(m) * n) {
        fill_random(host_A, -0.1f, 0.1f);
        fill_random(host_B, -0.1f, 0.1f);
        fill_random(host_C, -0.1f, 0.1f);

        device_A = host_A;
        device_B = host_B;
        device_C = host_C;
    }
};

template <class T>
bool compute_cublas_reference(
    cublasHandle_t handle,
    const GemmProblem<T>& problem,
    thrust::host_vector<T>& host_C_ref) {
    thrust::device_vector<T> device_C_ref(static_cast<size_t>(problem.M) * problem.N);
    if (!check_cublas(
            launch_cublas_gemm(
                handle,
                problem.M,
                problem.N,
                problem.K,
                problem.device_A.data().get(),
                problem.device_B.data().get(),
                device_C_ref.data().get()),
            "cublasGemmEx",
            __FILE__,
            __LINE__)) {
        return false;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__)) {
        return false;
    }

    host_C_ref = device_C_ref;
    return true;
}

template <class T>
bool benchmark_cublas(
    cudaStream_t stream,
    cublasHandle_t handle,
    GemmProblem<T>& problem,
    float& latency_ms) {
    return benchmark_launcher(
        stream,
        kWarmupIterations,
        kBenchmarkIterations,
        [&]() {
            return check_cublas(
                launch_cublas_gemm(
                    handle,
                    problem.M,
                    problem.N,
                    problem.K,
                    problem.device_A.data().get(),
                    problem.device_B.data().get(),
                    problem.device_C.data().get()),
                "cublasGemmEx",
                __FILE__,
                __LINE__);
        },
        latency_ms);
}

bool config_supports_problem(const GemmConfig& config, int M, int N, int K) {
    return M % config.bm == 0 && N % config.bn == 0 && K % config.bk == 0;
}

void print_config_skipped(const char* mode, const GemmConfig& config, int M, int N, int K) {
    std::printf(
        "%s skipped: %s M=%d N=%d K=%d requires "
        "M%%BM==0 N%%BN==0 K%%BK==0 (BM=%d BN=%d BK=%d)\n",
        mode,
        config.name,
        M,
        N,
        K,
        config.bm,
        config.bn,
        config.bk);
}

bool matches_config_name(const GemmConfig& config, std::string_view query) {
    const std::string_view name(config.name);
    constexpr std::string_view alias_prefix = "gemm_tn_";

    if (query == name) {
        return true;
    }
    return query.starts_with(alias_prefix) && query.substr(alias_prefix.size()) == name;
}

bool config_name_has_prefix(const GemmConfig& config, std::string_view prefix) {
    const std::string_view name(config.name);
    constexpr std::string_view alias_prefix = "gemm_tn_";

    if (name.starts_with(prefix)) {
        return true;
    }
    if (prefix.size() <= alias_prefix.size()) {
        return alias_prefix.starts_with(prefix);
    }
    return prefix.starts_with(alias_prefix) && name.starts_with(prefix.substr(alias_prefix.size()));
}

const GemmConfig* find_config(std::string_view query) {
    const bool is_prefix_pattern = !query.empty() && query.back() == '*';
    const std::string_view prefix = is_prefix_pattern ? query.substr(0, query.size() - 1) : query;

    const GemmConfig* match = nullptr;
    size_t match_count = 0;
    for (size_t i = 0; i < kGemmConfigCount; ++i) {
        const bool matched = is_prefix_pattern
            ? config_name_has_prefix(kGemmConfigs[i], prefix)
            : matches_config_name(kGemmConfigs[i], query);
        if (!matched) {
            continue;
        }
        match = &kGemmConfigs[i];
        ++match_count;
    }

    if (match_count == 1) {
        return match;
    }
    if (match_count > 1) {
        std::fprintf(stderr, "ambiguous config pattern: %.*s (%zu matches)\n",
                     static_cast<int>(query.size()), query.data(), match_count);
    } else {
        std::fprintf(stderr, "unknown config: %.*s\n", static_cast<int>(query.size()), query.data());
    }
    return nullptr;
}

void print_available_configs() {
    std::fprintf(stderr, "available configs:\n");
    for (size_t i = 0; i < kGemmConfigCount; ++i) {
        std::fprintf(stderr, "  gemm_tn_%s\n", kGemmConfigs[i].name);
    }
}

void print_usage(const char* program) {
    std::fprintf(
        stderr,
        "usage: %s --verify|--benchmark|--benchmark m n k|m n k"
        "|--func <gemm_tn_config_name> [--verify|--benchmark]"
        "|--profile <gemm_tn_config_name> m n k [warmup]\n",
        program);
}

std::string dirname_of(std::string path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

std::string executable_directory(const char* program) {
    char exe_path[PATH_MAX + 1] = {};
    const ssize_t len = readlink("/proc/self/exe", exe_path, PATH_MAX);
    if (len > 0) {
        exe_path[len] = '\0';
        return dirname_of(exe_path);
    }

    if (program == nullptr || program[0] == '\0') {
        return ".";
    }
    return dirname_of(program);
}

std::string best_output_path(const char* program) {
    const std::string dir = executable_directory(program);
    if (dir == "/") {
        return "/BEST.txt";
    }
    return dir + "/BEST.txt";
}

bool parse_positive_int_arg(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void print_verify_status(const GemmShape& shape, const char* status, const GemmConfig& config) {
    print_shape_braced(shape);
    std::printf(" verify %s: %s\n", status, config.name);
}

template <class T>
bool verify_one_config(
    const GemmConfig& config,
    int M,
    int N,
    int K,
    const thrust::device_vector<T>& device_A,
    const thrust::device_vector<T>& device_B,
    thrust::device_vector<T>& device_C,
    const thrust::host_vector<T>& host_C_init,
    const thrust::host_vector<T>& host_C_ref,
    bool print_detail) {
    device_C = host_C_init;
    config.run(
        M,
        N,
        K,
        device_A.data().get(),
        device_B.data().get(),
        device_C.data().get(),
        nullptr);

    bool passed = check_cuda(cudaGetLastError(), config.name, __FILE__, __LINE__);
    if (passed) {
        passed = check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
    if (passed) {
        thrust::host_vector<T> host_C_gemm = device_C;
        passed = verify_result(host_C_gemm, host_C_ref, N, kAbsTolerance, kRelTolerance, print_detail);
    }

    return passed;
}

template <class T>
bool verify_single_config(const GemmConfig& config, GemmProblem<T>& problem, bool print_success) {
    cublasHandle_t handle = nullptr;
    if (!check_cublas(cublasCreate(&handle), "cublasCreate(&handle)", __FILE__, __LINE__)) {
        return false;
    }

    thrust::host_vector<T> host_C_ref;
    const bool reference_ok = compute_cublas_reference(handle, problem, host_C_ref);
    const bool destroy_ok = check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
    if (!reference_ok || !destroy_ok) {
        return false;
    }

    const GemmShape shape{problem.M, problem.N, problem.K};
    const bool passed = verify_one_config(
        config,
        problem.M,
        problem.N,
        problem.K,
        problem.device_A,
        problem.device_B,
        problem.device_C,
        problem.host_C,
        host_C_ref,
        false);
    if (passed) {
        if (print_success) {
            print_verify_status(shape, "success", config);
        }
    } else {
        print_verify_status(shape, "failed", config);
    }
    return passed;
}

template <class T>
VerifySummary verify_all_configs_for_shape(
    cublasHandle_t handle,
    const GemmShape& shape,
    bool print_success,
    bool print_skipped) {
    VerifySummary summary;
    summary.total = kGemmConfigCount;
    GemmProblem<T> problem(shape.M, shape.N, shape.K);

    thrust::host_vector<T> host_C_ref;
    if (!compute_cublas_reference(handle, problem, host_C_ref)) {
        summary.ok = false;
        summary.failed = summary.total;
        print_shape_braced(shape);
        std::printf(
            " verify summary: passed=%zu failed=%zu skipped=%zu total=%zu\n",
            summary.passed,
            summary.failed,
            summary.skipped,
            summary.total);
        return summary;
    }

    for (size_t i = 0; i < kGemmConfigCount; ++i) {
        if (!config_supports_problem(kGemmConfigs[i], shape.M, shape.N, shape.K)) {
            if (print_skipped) {
                print_verify_status(shape, "skipped", kGemmConfigs[i]);
            }
            ++summary.skipped;
            continue;
        }
        const bool passed = verify_one_config(
                kGemmConfigs[i],
                shape.M,
                shape.N,
                shape.K,
                problem.device_A,
                problem.device_B,
                problem.device_C,
                problem.host_C,
                host_C_ref,
                false);
        if (passed) {
            ++summary.passed;
            if (print_success) {
                print_verify_status(shape, "success", kGemmConfigs[i]);
            }
        } else {
            ++summary.failed;
            print_verify_status(shape, "failed", kGemmConfigs[i]);
        }
    }

    summary.failed = summary.total - summary.passed - summary.skipped;
    print_shape_braced(shape);
    std::printf(" verify summary: passed=%zu failed=%zu skipped=%zu total=%zu\n",
                summary.passed,
                summary.failed,
                summary.skipped,
                summary.total);
    return summary;
}

template <class T>
int run_verify() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));

    bool all_ok = true;
    for (const GemmShape& shape : kTestShapes) {
        const VerifySummary summary = verify_all_configs_for_shape<T>(handle, shape, false, false);
        all_ok = all_ok && summary.ok && summary.failed == 0;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

void print_benchmark_line(
    size_t rank,
    const GemmConfig& config,
    const BenchmarkResult& result,
    float cublas_latency_ms,
    double cublas_throughput) {
    std::printf(
        "%4zu. %s %.4f ms %.3f TFLOP/s cublas %.4f ms %.3f TFLOP/s gemm_tn/cublas=%.2f%%\n",
        rank,
        config.name,
        result.latency_ms,
        result.throughput,
        cublas_latency_ms,
        cublas_throughput,
        throughput_percent(result.throughput, cublas_throughput));
}

void print_benchmark_result(
    const GemmConfig& config,
    const BenchmarkResult& result,
    float cublas_latency_ms,
    double cublas_throughput) {
    std::printf(
        "%s %.4f ms %.3f TFLOP/s cublas %.4f ms %.3f TFLOP/s gemm_tn/cublas=%.2f%%\n",
        config.name,
        result.latency_ms,
        result.throughput,
        cublas_latency_ms,
        cublas_throughput,
        throughput_percent(result.throughput, cublas_throughput));
}

std::string format_best_line(
    const GemmShape& shape,
    const GemmConfig& config,
    const BenchmarkResult& result,
    float cublas_latency_ms,
    double cublas_throughput) {
    char line[1024];
    std::snprintf(
        line,
        sizeof(line),
        "BEST: %d,%d,%d %s   %.4f ms %.3f TFLOP/s cublas %.4f ms %.3f TFLOP/s gemm_tn/cublas=%.2f%%\n",
        shape.M,
        shape.N,
        shape.K,
        config.name,
        result.latency_ms,
        result.throughput,
        cublas_latency_ms,
        cublas_throughput,
        throughput_percent(result.throughput, cublas_throughput));
    return line;
}

void print_best_line(
    const GemmShape& shape,
    const GemmConfig& config,
    const BenchmarkResult& result,
    float cublas_latency_ms,
    double cublas_throughput) {
    const std::string line = format_best_line(shape, config, result, cublas_latency_ms, cublas_throughput);
    std::fputs(line.c_str(), stdout);
}

template <class T>
BenchmarkResult benchmark_one_config(
    const GemmConfig& config,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    const thrust::device_vector<T>& device_A,
    const thrust::device_vector<T>& device_B,
    thrust::device_vector<T>& device_C,
    bool print_result,
    float cublas_latency_ms,
    double cublas_throughput) {
    float gemm_tn_latency_ms = 0.0f;
    const bool ok = benchmark_launcher(
        stream,
        kWarmupIterations,
        kBenchmarkIterations,
        [&]() {
            config.run(
                M,
                N,
                K,
                device_A.data().get(),
                device_B.data().get(),
                device_C.data().get(),
                stream);
            return check_cuda(cudaGetLastError(), config.name, __FILE__, __LINE__);
        },
        gemm_tn_latency_ms);

    if (!ok) {
        if (print_result) {
            std::printf(
                "benchmark failed: %s cublas %.4f ms %.3f TFLOP/s\n",
                config.name,
                cublas_latency_ms,
                cublas_throughput);
        }
        return {};
    }

    const double gemm_tn_throughput = gemm_tflops(M, N, K, gemm_tn_latency_ms);
    BenchmarkResult result;
    result.ok = true;
    result.latency_ms = gemm_tn_latency_ms;
    result.throughput = gemm_tn_throughput;
    if (print_result) {
        print_benchmark_result(config, result, cublas_latency_ms, cublas_throughput);
    }
    return result;
}

template <class T>
BenchmarkResult benchmark_single_config(
    const GemmConfig& config,
    GemmProblem<T>& problem,
    bool print_result) {
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)",
                    __FILE__,
                    __LINE__)) {
        return {};
    }

    cublasHandle_t handle = nullptr;
    if (!check_cublas(cublasCreate(&handle), "cublasCreate(&handle)", __FILE__, __LINE__)) {
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        return {};
    }
    if (!check_cublas(cublasSetStream(handle, stream), "cublasSetStream(handle, stream)", __FILE__, __LINE__)) {
        check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        return {};
    }

    float cublas_latency_ms = 0.0f;
    const bool cublas_ok = benchmark_cublas(stream, handle, problem, cublas_latency_ms);
    if (!cublas_ok) {
        check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        return {};
    }

    const double cublas_throughput = gemm_tflops(problem.M, problem.N, problem.K, cublas_latency_ms);
    BenchmarkResult result = benchmark_one_config(
        config,
        problem.M,
        problem.N,
        problem.K,
        stream,
        problem.device_A,
        problem.device_B,
        problem.device_C,
        print_result,
        cublas_latency_ms,
        cublas_throughput);

    const bool destroy_handle_ok = check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
    const bool destroy_stream_ok = check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
    if (!destroy_handle_ok || !destroy_stream_ok) {
        return {};
    }
    return result;
}

void print_benchmark_ranking(
    std::vector<BenchmarkSummary>& summaries,
    float cublas_latency_ms,
    double cublas_throughput) {
    std::stable_sort(
        summaries.begin(),
        summaries.end(),
        [](const BenchmarkSummary& lhs, const BenchmarkSummary& rhs) {
            if (lhs.result.ok != rhs.result.ok) {
                return lhs.result.ok;
            }
            if (lhs.result.skipped != rhs.result.skipped) {
                return !lhs.result.skipped;
            }
            if (!lhs.result.ok) {
                return lhs.original_index < rhs.original_index;
            }
            if (lhs.result.throughput != rhs.result.throughput) {
                return lhs.result.throughput > rhs.result.throughput;
            }
            return lhs.result.latency_ms < rhs.result.latency_ms;
        });

    size_t rank = 0;
    for (size_t i = 0; i < summaries.size(); ++i) {
        const BenchmarkSummary& summary = summaries[i];
        if (summary.result.skipped) {
            continue;
        }
        ++rank;
        if (!summary.result.ok) {
            std::printf("%4zu. %s FAILED\n", rank, summary.config->name);
            continue;
        }
        print_benchmark_line(rank, *summary.config, summary.result, cublas_latency_ms, cublas_throughput);
    }
}

void print_benchmark_best(
    const GemmShape& shape,
    const GemmConfig& config,
    const BenchmarkResult& result,
    float cublas_latency_ms,
    double cublas_throughput) {
    print_best_line(shape, config, result, cublas_latency_ms, cublas_throughput);
}

bool write_best_results_file(const char* program, const std::vector<BenchmarkBest>& best_results) {
    const std::string path = best_output_path(program);
    FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "failed to open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    bool ok = true;
    for (const BenchmarkBest& best : best_results) {
        const std::string line = format_best_line(
            best.shape,
            *best.config,
            best.result,
            best.cublas_latency_ms,
            best.cublas_throughput);
        if (std::fputs(line.c_str(), file) == EOF) {
            std::fprintf(stderr, "failed to write %s: %s\n", path.c_str(), std::strerror(errno));
            ok = false;
            break;
        }
    }

    if (std::fclose(file) != 0) {
        std::fprintf(stderr, "failed to close %s: %s\n", path.c_str(), std::strerror(errno));
        ok = false;
    }
    return ok;
}

template <class T>
BenchmarkShapeResult benchmark_all_configs_for_shape(const GemmShape& shape) {
    BenchmarkShapeResult shape_result;
    shape_result.total = kGemmConfigCount;
    shape_result.best.shape = shape;
    GemmProblem<T> problem(shape.M, shape.N, shape.K);

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)",
                    __FILE__,
                    __LINE__)) {
        shape_result.ok = false;
        shape_result.failed = shape_result.total;
        return shape_result;
    }

    cublasHandle_t handle = nullptr;
    if (!check_cublas(cublasCreate(&handle), "cublasCreate(&handle)", __FILE__, __LINE__)) {
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        shape_result.ok = false;
        shape_result.failed = shape_result.total;
        return shape_result;
    }
    if (!check_cublas(cublasSetStream(handle, stream), "cublasSetStream(handle, stream)", __FILE__, __LINE__)) {
        check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        shape_result.ok = false;
        shape_result.failed = shape_result.total;
        return shape_result;
    }

    float cublas_latency_ms = 0.0f;
    const bool cublas_ok = benchmark_cublas(stream, handle, problem, cublas_latency_ms);
    if (!cublas_ok) {
        check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
        shape_result.ok = false;
        shape_result.failed = shape_result.total;
        return shape_result;
    }

    const double cublas_throughput = gemm_tflops(shape.M, shape.N, shape.K, cublas_latency_ms);
    shape_result.best.cublas_latency_ms = cublas_latency_ms;
    shape_result.best.cublas_throughput = cublas_throughput;
    std::vector<BenchmarkSummary> summaries;
    summaries.reserve(kGemmConfigCount);
    for (size_t i = 0; i < kGemmConfigCount; ++i) {
        if (!config_supports_problem(kGemmConfigs[i], shape.M, shape.N, shape.K)) {
            BenchmarkResult skipped_result;
            skipped_result.skipped = true;
            summaries.push_back(BenchmarkSummary{&kGemmConfigs[i], skipped_result, i});
            ++shape_result.skipped;
            continue;
        }
        const BenchmarkResult result = benchmark_one_config(
                kGemmConfigs[i],
                shape.M,
                shape.N,
                shape.K,
                stream,
                problem.device_A,
                problem.device_B,
                problem.device_C,
                false,
                cublas_latency_ms,
                cublas_throughput);
        summaries.push_back(BenchmarkSummary{&kGemmConfigs[i], result, i});
        if (result.ok) {
            ++shape_result.passed;
            if (shape_result.best.config == nullptr || result.throughput > shape_result.best.result.throughput) {
                shape_result.best.config = &kGemmConfigs[i];
                shape_result.best.result = result;
            }
        }
    }

    const bool destroy_handle_ok = check_cublas(cublasDestroy(handle), "cublasDestroy(handle)", __FILE__, __LINE__);
    const bool destroy_stream_ok = check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
    shape_result.ok = destroy_handle_ok && destroy_stream_ok;
    shape_result.failed = shape_result.total - shape_result.passed - shape_result.skipped;

    print_shape_braced(shape);
    std::printf(" benchmark\n");
    print_benchmark_ranking(summaries, cublas_latency_ms, cublas_throughput);
    return shape_result;
}

template <class T>
int run_benchmark(const char* program) {
    bool all_ok = true;
    std::vector<BenchmarkBest> best_results;
    best_results.reserve(sizeof(kTestShapes) / sizeof(kTestShapes[0]));

    for (const GemmShape& shape : kTestShapes) {
        BenchmarkShapeResult shape_result = benchmark_all_configs_for_shape<T>(shape);
        all_ok = all_ok && shape_result.ok && shape_result.failed == 0 && shape_result.best.config != nullptr;
        if (shape_result.best.config != nullptr) {
            best_results.push_back(shape_result.best);
        }
    }

    for (const BenchmarkBest& best : best_results) {
        print_benchmark_best(
            best.shape,
            *best.config,
            best.result,
            best.cublas_latency_ms,
            best.cublas_throughput);
    }
    all_ok = write_best_results_file(program, best_results) && all_ok;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

template <class T>
int run_func(std::string_view query, FuncMode mode) {
    constexpr GemmShape shape{4096, 4096, 4096};
    const GemmConfig* config = find_config(query);
    if (config == nullptr) {
        print_available_configs();
        return EXIT_FAILURE;
    }
    if (!config_supports_problem(*config, shape.M, shape.N, shape.K)) {
        print_config_skipped("func", *config, shape.M, shape.N, shape.K);
        return EXIT_FAILURE;
    }

    std::printf(
        "func config=%s mode=%s M=%d N=%d K=%d warmup=%d iterations=%d\n",
        config->name,
        func_mode_name(mode),
        shape.M,
        shape.N,
        shape.K,
        kWarmupIterations,
        kBenchmarkIterations);

    GemmProblem<T> problem(shape.M, shape.N, shape.K);
    bool verify_ok = true;
    bool benchmark_ok = true;

    if (mode != FuncMode::BenchmarkOnly) {
        verify_ok = verify_single_config(*config, problem, true);
    }
    if (mode != FuncMode::VerifyOnly) {
        const BenchmarkResult benchmark_result = benchmark_single_config(*config, problem, true);
        benchmark_ok = benchmark_result.ok;
    }

    return verify_ok && benchmark_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Launch one config on one shape with a single kernel launch fenced inside a
// cudaProfilerStart/Stop region (after warmup launches), then exit. This
// isolates exactly one launch for Nsight Compute via `--profile-from-start off
// -c 1`, mirroring flashattention/v3_mMnNkN/test_compute_rate.py.
template <class T>
int run_profile(std::string_view query, const GemmShape& shape, int warmup) {
    const GemmConfig* config = find_config(query);
    if (config == nullptr) {
        print_available_configs();
        return EXIT_FAILURE;
    }
    if (!config_supports_problem(*config, shape.M, shape.N, shape.K)) {
        print_config_skipped("profile", *config, shape.M, shape.N, shape.K);
        return EXIT_FAILURE;
    }

    std::printf(
        "profile config=%s M=%d N=%d K=%d warmup=%d\n",
        config->name,
        shape.M,
        shape.N,
        shape.K,
        warmup);

    GemmProblem<T> problem(shape.M, shape.N, shape.K);
    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)",
                    __FILE__,
                    __LINE__)) {
        return EXIT_FAILURE;
    }

    auto launch = [&]() {
        config->run(
            shape.M,
            shape.N,
            shape.K,
            problem.device_A.data().get(),
            problem.device_B.data().get(),
            problem.device_C.data().get(),
            stream);
        return check_cuda(cudaGetLastError(), config->name, __FILE__, __LINE__);
    };

    bool ok = true;
    for (int i = 0; i < warmup && ok; ++i) {
        ok = launch();
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize()", __FILE__, __LINE__);

    if (ok) {
        cudaProfilerStart();
        ok = launch();
        ok = ok && check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize()", __FILE__, __LINE__);
        cudaProfilerStop();
    }

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(stream)", __FILE__, __LINE__);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Benchmark every config on one shape and print the best, skipping the CPU-side
// verification. Useful for large shapes where the host reference compare would
// dominate runtime (e.g. 16384^3).
template <class T>
int run_shape_benchmark(const GemmShape& shape) {
    BenchmarkShapeResult benchmark_result = benchmark_all_configs_for_shape<T>(shape);
    if (benchmark_result.best.config != nullptr) {
        print_benchmark_best(
            benchmark_result.best.shape,
            *benchmark_result.best.config,
            benchmark_result.best.result,
            benchmark_result.best.cublas_latency_ms,
            benchmark_result.best.cublas_throughput);
    }
    return benchmark_result.ok && benchmark_result.failed == 0 &&
                   benchmark_result.best.config != nullptr
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

template <class T>
int run_shape_test(const GemmShape& shape) {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    const VerifySummary verify_summary = verify_all_configs_for_shape<T>(handle, shape, true, true);
    CHECK_CUBLAS(cublasDestroy(handle));

    BenchmarkShapeResult benchmark_result = benchmark_all_configs_for_shape<T>(shape);
    if (benchmark_result.best.config != nullptr) {
        print_benchmark_best(
            benchmark_result.best.shape,
            *benchmark_result.best.config,
            benchmark_result.best.result,
            benchmark_result.best.cublas_latency_ms,
            benchmark_result.best.cublas_throughput);
    }

    return verify_summary.ok && verify_summary.failed == 0 &&
                   benchmark_result.ok && benchmark_result.failed == 0 &&
                   benchmark_result.best.config != nullptr
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    using T = bf16;

    if (argc == 2 && std::strcmp(argv[1], "--verify") == 0) {
        return run_verify<T>();
    }
    if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
        return run_benchmark<T>(argv[0]);
    }
    if (argc == 5 && std::strcmp(argv[1], "--benchmark") == 0) {
        GemmShape shape{};
        if (!parse_positive_int_arg(argv[2], shape.M) ||
            !parse_positive_int_arg(argv[3], shape.N) ||
            !parse_positive_int_arg(argv[4], shape.K)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_shape_benchmark<T>(shape);
    }
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--func") == 0) {
        FuncMode mode = FuncMode::VerifyAndBenchmark;
        if (argc == 4) {
            if (std::strcmp(argv[3], "--verify") == 0) {
                mode = FuncMode::VerifyOnly;
            } else if (std::strcmp(argv[3], "--benchmark") == 0) {
                mode = FuncMode::BenchmarkOnly;
            } else {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        return run_func<T>(argv[2], mode);
    }
    if ((argc == 6 || argc == 7) && std::strcmp(argv[1], "--profile") == 0) {
        GemmShape shape{};
        if (!parse_positive_int_arg(argv[3], shape.M) ||
            !parse_positive_int_arg(argv[4], shape.N) ||
            !parse_positive_int_arg(argv[5], shape.K)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        int warmup = 3;
        if (argc == 7 && !parse_positive_int_arg(argv[6], warmup)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_profile<T>(argv[2], shape, warmup);
    }
    if (argc == 4) {
        GemmShape shape{};
        if (!parse_positive_int_arg(argv[1], shape.M) ||
            !parse_positive_int_arg(argv[2], shape.N) ||
            !parse_positive_int_arg(argv[3], shape.K)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_shape_test<T>(shape);
    }

    print_usage(argv[0]);
    return EXIT_FAILURE;
}

/*
test_gemm_cute execution notes.

Build output:
  The executable is produced by the CMake target test_gemm_cute. With the
  commands documented in CMakeLists.txt, typical executable paths are:
    ./build/test_gemm_cute
    ./build_one/test_gemm_cute

Common run path:
  cd gemm

Supported command lines:
  1. Verify every generated gemm_tn config on all built-in shapes:
       ./build/test_gemm_cute --verify

  2. Benchmark every generated gemm_tn config on all built-in shapes:
       ./build/test_gemm_cute --benchmark
     This also writes BEST.txt next to the executable, for example:
       ./build/BEST.txt

  3. Verify and benchmark one config on the fixed func shape M=N=K=4096:
       ./build/test_gemm_cute --func <config>

  4. Verify only one config on the fixed func shape M=N=K=4096:
       ./build/test_gemm_cute --func <config> --verify

  5. Benchmark only one config on the fixed func shape M=N=K=4096:
       ./build/test_gemm_cute --func <config> --benchmark

  6. Verify and benchmark every generated config on one custom shape:
       ./build/test_gemm_cute <M> <N> <K>
     M, N and K must be positive int values.

Config name arguments:
  <config> is a generated gemm_tn config name. The "gemm_tn_" prefix is
  optional, and a trailing "*" is accepted as a prefix pattern if it matches
  exactly one config.

  Examples:
    ./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4
    ./build/test_gemm_cute --func gemm_tn_bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4
    ./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4 --verify
    ./build/test_gemm_cute --func bf16_bm128_bn64_bk64_s2_cwg2_wm2_wn4_bsw4 --benchmark

Generated config parameter fields:
  bf16_bm<BM>_bn<BN>_bk<BK>_s<STAGE>_cwg<CWG>_wm<WARPS_M>_wn<WARPS_N>_bsw<BS_W>

  Values are generated by gemm_configs.py and filtered by its legality checks.
  Current generator inputs are:
    type=bf16
    BM={64,128,256}
    BN={64,128,256}
    BK={64,128}
    STAGE={2,3}
    CWG={2,4}
    BS_W={4,8}
    WARPS_M/WARPS_N from legal_warp_shapes(), with WARPS_M*WARPS_N=4*CWG.

Runtime filtering:
  A config only runs when M % BM == 0, N % BN == 0 and K % BK == 0.
  Unsupported configs are reported as skipped.

Built-in shapes used by --verify and --benchmark:
  {128, 4096, 14336}
  {128, 28672, 4096}
  {512, 512, 14336}
  {1024, 1024, 1024}
  {1024, 1024, 14336}
  {2048, 2048, 2048}
  {4096, 4096, 4096}
  {4096, 4096, 14336}
  {4096, 28672, 4096}
  {8192, 8192, 8192}
  {16384, 16384, 16384}
  {40960, 40960, 4096}

Benchmark and verification constants:
  Data type: bf16
  Warmup iterations: 5
  Benchmark iterations: 25
  Verification tolerance: abs=1.0e-2, rel=5.0e-2
*/
