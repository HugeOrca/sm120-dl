#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "flash_fwd_config_registry.hpp"

namespace {

using Shape = std::array<int64_t, 4>;

std::vector<Shape> supported_shapes() {
    return {
        Shape{64, 64, 1, 1},
        Shape{128, 64, 1, 1},
        Shape{128, 64, 1, 2},
        Shape{192, 64, 1, 1},
        Shape{128, 64, 4, 1},
        Shape{192, 64, 4, 1},
        Shape{256, 64, 8, 2},
        Shape{128, 128, 4, 1},
        Shape{128, 128, 4, 2},
        Shape{256, 128, 4, 1},
        Shape{2048, 64, 16, 1},
        Shape{4096, 128, 32, 1},
        Shape{8192, 128, 32, 1},
        Shape{16384, 128, 32, 1},
        Shape{2048, 64, 32, 8},
        Shape{4096, 128, 64, 2},
        Shape{4096, 128, 64, 8},
        Shape{8192, 128, 80, 4},
    };
}

void check_input_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must use torch.float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [batch, seq_len, head_num, head_dim]");
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    check_input_tensor(q, "q");
    check_input_tensor(k, "k");
    check_input_tensor(v, "v");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q shape");
    TORCH_CHECK(k.device() == q.device(), "k device must match q device");
    TORCH_CHECK(v.device() == q.device(), "v device must match q device");
}

const FlashFwdConfig* find_config(const std::string& name) {
    for (std::size_t i = 0; i < kFlashFwdConfigCount; ++i) {
        if (name == kFlashFwdConfigs[i].name) {
            return &kFlashFwdConfigs[i];
        }
    }
    return nullptr;
}

const FlashFwdConfig& default_config(int64_t head_dim) {
    const FlashFwdConfig* config = find_config(kFlashFwdDefaultConfigName);
    if (config != nullptr && config->head_dim == head_dim) {
        return *config;
    }

    const int64_t preferred_block_m = head_dim == 64 ? 128 : 64;
    const std::string preferred = "hd" + std::to_string(head_dim) + "_bm" + std::to_string(preferred_block_m) + "_bn32_w8_s2";
    config = find_config(preferred);
    if (config != nullptr) {
        return *config;
    }

    for (std::size_t i = 0; i < kFlashFwdConfigCount; ++i) {
        if (kFlashFwdConfigs[i].head_dim == head_dim) {
            return kFlashFwdConfigs[i];
        }
    }
    TORCH_CHECK(false, "no flash_fwd config was built for head_dim=", head_dim);
}

void check_runtime_shape(const FlashFwdConfig& config, const torch::Tensor& q) {
    const int64_t batch = q.size(0);
    const int64_t seq_len = q.size(1);

    TORCH_CHECK(seq_len > 0, "seq_len must be positive");
    TORCH_CHECK(seq_len % config.block_m == 0, "seq_len must be a multiple of ", config.block_m);
    TORCH_CHECK(seq_len % config.block_n == 0, "seq_len must be a multiple of ", config.block_n);
    TORCH_CHECK(batch > 0, "batch must be positive");
}

torch::Tensor fwd_config(const std::string& config_name, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, bool causal) {
    check_inputs(q, k, v);
    const FlashFwdConfig* config = find_config(config_name);
    TORCH_CHECK(config != nullptr, "unknown flash_fwd config: ", config_name);
    check_runtime_shape(*config, q);
    return config->run(q, k, v, causal);
}

torch::Tensor fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, bool causal) {
    check_inputs(q, k, v);
    const FlashFwdConfig& config = default_config(q.size(3));
    check_runtime_shape(config, q);
    return config.run(q, k, v, causal);
}

std::vector<std::tuple<std::string, int, int, int, int, int, int>> configs() {
    std::vector<std::tuple<std::string, int, int, int, int, int, int>> result;
    result.reserve(kFlashFwdConfigCount);
    for (std::size_t i = 0; i < kFlashFwdConfigCount; ++i) {
        const auto& config = kFlashFwdConfigs[i];
        result.emplace_back(config.name, config.head_dim, config.block_m, config.block_n, config.nwarps, config.stage, config.smem_bytes);
    }
    return result;
}

}

PYBIND11_MODULE(flashatten_v3_sm120, m) {
    m.doc() = "pybind wrapper for flashattention/v3_mMn1k1/flashatten_v3.hpp";
    m.def("fwd", &fwd, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("causal") = false);
    m.def("fwd_config", &fwd_config, pybind11::arg("config"), pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("causal") = false);
    m.def("supported_shapes", &supported_shapes);
    m.def("configs", &configs);
    m.def("default_config_name", [] { return std::string(kFlashFwdDefaultConfigName); });
}
