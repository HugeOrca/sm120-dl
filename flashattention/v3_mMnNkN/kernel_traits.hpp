// SPDX-FileCopyrightText: 2024 Tri Dao
// SPDX-License-Identifier: BSD-3-Clause
// Adapted for this project; see README "Acknowledgements" and THIRD_PARTY_LICENSES.md.
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;

template<typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
};

template<int Stage>
struct PipelineStorage {
    alignas(16) uint64_t full_barrier[Stage];
    alignas(16) uint64_t empty_barrier[Stage];

    template<int FullCount, int EmptyCount>
    CUTLASS_DEVICE void init_barrier() {
        using Barrier = cutlass::arch::ClusterBarrier;
        using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
        if(threadIdx.x < Stage) {
            FullBarrier::init(&(full_barrier[threadIdx.x]),  FullCount);
            Barrier::init(&(empty_barrier[threadIdx.x]), EmptyCount);
        }
    }
};

template<class ValType,
          int kBlockM, int kBlockN, int kHeadDim,
          int STAGE, int WARPS_N = 1>
struct Storage {
    static_assert(kHeadDim % 32 ==0,
        "kHeadDim must be a multiple of 32");
    static_assert(STAGE == 2, "QK smem reuse requires STAGE=2");
    using SW_K_ATOM = conditional_t<kHeadDim%64 == 0,
        typename GMMA::Layout_K_SW128_Atom<ValType>,
        typename GMMA::Layout_K_SW64_Atom<ValType>>;
    using LayoutQ = decltype(tile_to_shape(
        SW_K_ATOM{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using LayoutO = LayoutQ;
    using LayoutKStage = decltype(tile_to_shape(
        SW_K_ATOM{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    constexpr static int q_size = cute::cosize_v<LayoutQ>;
    constexpr static int k_size = cute::cosize_v<LayoutKStage>;
    constexpr static int o_size = q_size;
    constexpr static int v_size = k_size;
    constexpr static int v_stage = 1;
    constexpr static int o_slots = WARPS_N > 1 ? WARPS_N - 1 : 1;
    constexpr static int QK_size = k_size + (q_size >= k_size ? q_size : k_size);
    constexpr static int k_stage_stride = q_size <= k_size ? k_size : q_size;
    constexpr static int QKV_size = QK_size + v_size;
    constexpr static int O_size = o_slots * o_size;
    constexpr static int QKVO_size = QKV_size >= O_size ? QKV_size : O_size;
    constexpr static int row_stats_slots = WARPS_N > 1 ? WARPS_N * (kBlockM / 16) * 16 : 1;
    using LayoutK = decltype(cute::append<3>(
        LayoutKStage{},
        Layout<Int<STAGE>, Int<k_stage_stride>>{}));
    using LayoutV = LayoutKStage;
    using LayoutVtStage = decltype(composition(
        LayoutKStage{},
        Layout<Shape<Int<kHeadDim>, Int<kBlockN>>,
               Stride<Int<kBlockN>, _1>>{}));

    alignas(128) cute::array_aligned<ValType, QKVO_size> QKV;
    alignas(128) cute::array_aligned<float, row_stats_slots> row_max;
    alignas(128) cute::array_aligned<float, row_stats_slots> row_sum;
    PipelineStorage<STAGE> p_qk;
    PipelineStorage<v_stage> p_v;

    __forceinline__ __device__ auto get_sQ(int idx) {
        return make_tensor(make_smem_ptr(QKV.data() + k_size * idx), LayoutQ{});
    }
    __forceinline__ __device__ auto get_sK() {
        return make_tensor(make_smem_ptr(QKV.data()), LayoutK{});
    }
    __forceinline__ __device__ auto get_sV() {
        return make_tensor(make_smem_ptr(QKV.data() + QK_size), LayoutV{});
    }
    __forceinline__ __device__ auto get_sVt_S() {
        return make_tensor(make_smem_ptr(QKV.data() + QK_size), LayoutVtStage{});
    }
    __forceinline__ __device__ auto get_sVt_D() {
        return make_tensor(make_smem_ptr(QKV.data() + QK_size), LayoutVtStage{}.layout_b());
    }
    __forceinline__ __device__ auto get_sO(int idx = 0) {
        return make_tensor(make_smem_ptr(QKV.data() + idx * o_size), LayoutO{});
    }
};

template<int kHeadNum_, int kHeadDim_,
         int kBlockM_, int kBlockN_, int WARPS_M_, int WARPS_N_, int STAGE_,
         typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = elem_type;
    using ElementAccum  = typename Base::ElementAccum;

    constexpr static int WARPS_M  = WARPS_M_;
    constexpr static int WARPS_N  = WARPS_N_;
    constexpr static int kNWarps  = WARPS_M * WARPS_N;
    static constexpr int kNThreads= kNWarps * 32;
    constexpr static int kBlockM  = kBlockM_;
    constexpr static int kBlockN  = kBlockN_;
    constexpr static int STAGE    = STAGE_;

    constexpr static int kHeadNum = kHeadNum_;
    constexpr static int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);

    static_assert(kBlockM % (16 * WARPS_M) == 0 && kBlockN % (16 * WARPS_N) == 0);

    static constexpr auto EU = make_layout(make_shape(Int<WARPS_M>{}, Int<WARPS_N>{}));
    constexpr static int WM = kBlockM / (16 * WARPS_M);
    constexpr static int WN = kBlockN / (8 * WARPS_N);
    using Permute_QK = Tile<
        Layout<Shape<_16, Int<WARPS_M>, Int<WM>>, Stride<_1, Int<16 * WM>, _16>>,
        Layout<Shape<_8, Int<WARPS_N>, Int<WN>>, Stride<_1, Int<8 * WN>, _8>>,
        _16>;
    using TiledMMA_QK = decltype(make_tiled_mma(typename Base::MMA_Atom_Arch{}, EU, Permute_QK{}));

    constexpr static int WK = kBlockN / (16 * WARPS_N);
    using Permute_PV = Tile<
        Int<kBlockM>,
        Int<kHeadDim>,
        Layout<Shape<_16, Int<WARPS_N>, Int<WK>>, Stride<_1, Int<16 * WK>, _16>>>;
    using TiledMMA_PV = decltype(make_tiled_mma(
        typename Base::MMA_Atom_Arch{},
        Layout<Shape<Int<WARPS_M>, _1, Int<WARPS_N>>>{},
        Permute_PV{}));

    using SStorage =
        Storage<Element, kBlockM, kBlockN, kHeadDim, STAGE, WARPS_N>;

    template<typename SLayout, typename Ptr, typename CopyOp = SM90_TMA_LOAD>
    auto get_tma(Ptr g_ptr, int seq_len, int batch, CopyOp copy_op = {}) {
        auto shape_qv = make_shape(
            seq_len, Int<kHeadDim>{},
            Int<kHeadNum>{}, batch);
        auto stride_qv = make_stride(
            Int<kHeadNum * kHeadDim>{}, _1{},
            Int<kHeadDim>{}, seq_len * kHeadNum * kHeadDim);
        Tensor tensor = make_tensor(make_gmem_ptr(g_ptr), shape_qv, stride_qv);
        auto layout   = take<0, 2>(SLayout{});
        auto cta_tile = make_shape(size<0>(layout), size<1>(layout));
        Copy_Atom tma =  make_tma_atom(copy_op, tensor, layout, cta_tile);
        Tensor tma_tensor = tma.get_tma_tensor(shape_qv);
        return cute::make_tuple(tma, tma_tensor);
    }
};
