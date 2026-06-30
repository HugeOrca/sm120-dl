// SPDX-FileCopyrightText: 2024 Tri Dao
// SPDX-License-Identifier: BSD-3-Clause
// Adapted for this project; see README "Acknowledgements" and THIRD_PARTY_LICENSES.md.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/config.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_conversion.h"

#include "utils/tools.h"
#include "utils/utils.h"
#include "kernel_traits.hpp"
#include "pipeline.hpp"

#define dv_flag  __device__ __forceinline__

template<int RegCount>
dv_flag void dealloc_warpgroup_registers() {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
}

template<int RegCount>
dv_flag void alloc_warpgroup_registers() {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
}

dv_flag auto get_s2r_src_dst(auto& cpy_atom, auto& mma, auto& src, auto& dst, auto const& tiled_func, int tid){
    auto s2r = tiled_func(cpy_atom, mma).get_slice(tid);
    Tensor cpy_src = s2r.partition_S(src);
    Tensor cpy_dst = s2r.retile_D(dst);
    return cute::make_tuple(cpy_src, cpy_dst);
}

dv_flag void O_r2s(auto& cpy_atom, auto& mma, auto& src, auto& dst, int tid) {
    auto r2s = make_tiled_copy_C(cpy_atom, mma);
    auto thr_r2s = r2s.get_slice(tid);
    Tensor cpy_src = thr_r2s.retile_S(src);
    Tensor cpy_dst = thr_r2s.partition_D(dst);
    copy(r2s, cpy_src, cpy_dst);
    tma_store_fence();
}

dv_flag void O_s2g(auto const& tma_o, auto const& mO, auto const& cta_O,
                   auto& sO, int q_block, int head_id, int batch_id) {
    Tensor gO = local_tile(mO(_, _, head_id, batch_id), cta_O, make_coord(q_block, _0{}));
    auto [tma_gO, tma_sO] = tma_partition(tma_o, group_modes<0, 2>(sO), group_modes<0, 2>(gO));
    copy(tma_o, tma_sO, tma_gO);
    tma_store_arrive();
}

dv_flag auto get_rowcol(auto& tensor) {
    auto layout = tensor.layout();
    return make_tensor(tensor.data(), make_layout(
        make_layout(get<0, 1>(layout), get<1>(layout)),
        make_layout(get<0, 0>(layout), get<2>(layout))
    ));
}


template<int kBlockN, int kBlockM, int kNWarps>
dv_flag void causal_mask(auto& tensor, int q_block, int i) {
    int row_offset = q_block*kBlockM + (threadIdx.y - 4)*16 + threadIdx.x/4;
    int col_offset = i*kBlockN + (threadIdx.x%4)*2;

    using Shape = decltype(shape(tensor));
    constexpr int M0 = size<0, 0>(Shape{});
    constexpr int M1 = size<0, 1>(Shape{});
    constexpr int N0 = size<1, 0>(Shape{});
    constexpr int N1 = size<1, 1>(Shape{});
    constexpr int row_stride = 16*kNWarps;
    constexpr int col_stride = 8;

    #pragma unroll
    for(int m1=0; m1<M1; ++m1) {
        #pragma unroll
        for(int m0=0; m0<M0; ++m0) {
            int row_idx = row_offset+m1*row_stride+m0*8;
            #pragma unroll
            for(int n1=0; n1<N1; ++n1) {
                #pragma unroll
                for(int n0=0; n0<N0; ++n0) {
                    int col_idx = col_offset + n1 * col_stride + n0;
                    if(row_idx < col_idx) {
                        tensor(make_coord(m0, m1), make_coord(n0, n1)) = -INFINITY;
                    }
                }
            }
        }
    }
}

template<int kNWarps>
dv_flag void bar_sync() {
        cutlass::arch::NamedBarrier::sync(kNWarps<<5, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
}

template<int kNWarps>
dv_flag void tile_buffer_sync() {
    cutlass::arch::NamedBarrier::sync((kNWarps + 4) << 5, 0);
}

template<int kNWarps>
dv_flag void normalize_stats(auto& rS_max, auto& rS_sum, auto& rO_RC, float scale) {
    CUTE_UNROLL
    for(int r=0; r<size<0>(rO_RC); ++r) {
        float sum = flash::warp_reduce_sum<4>(rS_sum(r));
        float inv_sum = sum == 0.f ? 1.f : __fdividef(1.f, sum);
        CUTE_UNROLL
        for(int i=0; i<size<1>(rO_RC); ++i) {
            rO_RC(r, i) *= inv_sum;
        }
    }
}

template<class ValType>
dv_flag
auto convert_f2h(auto const& tensor) {
    auto tensor_f16 = make_tensor<ValType>(shape(tensor));
    cutlass::NumericConverter<ValType, float> converter;
    #pragma unroll
    for(int i=0; i<size(tensor); ++i) {
        tensor_f16(i) = converter(tensor(i));
    }
    return tensor_f16;
}

dv_flag
auto Convert_Areg(auto layout) {
    using X = Underscore;
    auto l = logical_divide(layout, Shape<X, X, Int<2>>{});
    return make_layout(
        make_layout(get<0,0>(l), get<0,1>(l), get<2,0>(l)),
        get<1>(l),  get<2,1>(l)
    );

}

template<typename T>
dv_flag 
void softmax_rescale(auto& score_RC, auto& score_max, auto& score_sum, 
    auto& out_RC, auto scale) {
    auto score_max_prev = make_fragment_like(score_max);
    copy(score_max, score_max_prev);
    flash::reduce_max<4>(score_RC, score_max);
    #pragma unroll
    for(int r=0; r<size<0>(score_max); ++r) {
        float prev = score_max_prev(r);
        float smax = score_max(r);
        float exp_scale = smax == -INFINITY ? 1.f : exp2f((prev - smax) * scale);
        score_sum(r) *= exp_scale;
        #pragma unroll
        for(int i=0; i<size<1>(out_RC); ++i) 
            out_RC(r, i) *= exp_scale;

        float max_scale = score_max(r) == -INFINITY? 0 : score_max(r)*scale;
        #pragma unroll
        for(int i=0; i<size<1>(score_RC); ++i) 
            score_RC(r, i) = exp2f(score_RC(r, i)*scale - max_scale);
    }
    flash::batch_reduce(score_RC, score_sum, cute::plus{});
}

template <class Layout>
dv_flag constexpr auto half_first_shape(Layout const& layout) {
  auto s = shape(layout);
  auto d = stride(layout);

  if constexpr (rank(decltype(s){}) == Int<1>{}) {
    return make_layout(s / Int<2>{}, d);
  } else {
    return make_layout(replace<0>(s, get<0>(s) / Int<2>{}), d);
  }
}

template<typename T, typename DstTensor>
dv_flag void
gemm_flash(auto& s2r_B, auto& sB, int tid,
           auto& rC, auto& rA,
           auto& tiled_mma, auto& pipe) {
    auto s2r_B_func = [] (auto& atom, auto& tiled_mma) {
        return make_tiled_copy_B(atom, tiled_mma);
    };

    ThrMMA thr = tiled_mma.get_thread_slice(tid);
    auto rB = thr.partition_fragment_B(DstTensor{});
    auto [src_B, dst_B] = get_s2r_src_dst(s2r_B, tiled_mma, sB, rB, s2r_B_func, tid);
    auto consume_pipe = [&] (auto pipe_const) {
        constexpr int pipe_idx = decltype(pipe_const)::value;
        constexpr int K = size<2>(rB);
        CUTE_UNROLL
        for(int i=0; i<K; ++i) {
            copy(s2r_B, src_B(_, _, i, pipe_idx), dst_B(_, _, i));
            cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i), rC);
        }
    };
    pipe.wait();
    switch (pipe.index()) {
        case 0: { consume_pipe(Int<0>{}); break; }
        case 1: { consume_pipe(Int<1>{}); break; }
        default: { assert(false); }
    }
    pipe.release();
}

template<typename T, typename DstTensor>
dv_flag void
gemm_flash_v(auto& sstorage, auto& s2r_B, int tid,
             auto& rC, auto& rA,
             auto& tiled_mma, auto& pipe) {
    auto s2r_B_func = [] (auto& atom, auto& tiled_mma) {
        return make_tiled_copy_B(atom, tiled_mma);
    };

    ThrMMA thr = tiled_mma.get_thread_slice(tid);
    auto rB = thr.partition_fragment_B(DstTensor{});
    auto sB = sstorage.get_sVt_S();
    auto [src_B, dst_B] = get_s2r_src_dst(s2r_B, tiled_mma, sB, rB, s2r_B_func, tid);

    pipe.wait();
    constexpr int K = size<2>(rB);
    CUTE_UNROLL
    for(int n=0; n<K; ++n) {
        copy(s2r_B, src_B(_, _, n), dst_B(_, _, n));
        cute::gemm(tiled_mma, rA(_, _, n), rB(_, _, n), rC);
    }
    pipe.release();
}

dv_flag void Q_s2r(auto& sstorage, auto& pipe, auto& cpy_atom, auto& mma,  auto& dst, int tid) {
    Tensor sQ = sstorage.get_sQ(pipe.index());
    auto s2r_A_func = [] (auto& atom, auto& tiled_mma) {
        return make_tiled_copy_A(atom, tiled_mma);
    };
    auto [src_q, dst_q] = get_s2r_src_dst(cpy_atom, mma, sQ, dst, s2r_A_func, tid);
    pipe.wait();
    copy(cpy_atom, src_q, dst_q);
    pipe.release();
}

template<uint32_t Bytes>
dv_flag void Q_g2s(const auto& tma, const auto& mTensor, const auto& cta, const auto& coord, auto& sTensor, auto& pipe) {
    auto gTensor = local_tile(mTensor, cta, coord);
    auto [tma_gQ, tma_sQ] = tma_partition(tma, group_modes<0, 2>(sTensor), group_modes<0, 2>(gTensor));
    pipe.template copy_g2s<Bytes>(tma, tma_gQ, tma_sQ);
}

template<int kHeadNum>
dv_flag auto get_tile_coord(int M_num, int tiled_id) {
    int q_block    = tiled_id % M_num;
    q_block = M_num - 1 - q_block;
    int head_total = tiled_id / M_num;
    int head_id    = head_total % kHeadNum;
    int batch_id   = head_total / kHeadNum;
    return cute::make_coord(q_block, head_id, batch_id);
};

template<class T, bool is_causal, int kNWarps, class CONFIG_PARAMS,
class TensorQ, class TensorK, class TensorV, class TensorO,
class TmaQ, class TmaK, class TmaV, class TmaO>
__global__ void 
__launch_bounds__((4+kNWarps)*32, 1)
flash_atten_v3_fwd(    
    float scale,
    int seq_len,
    int batch,
    TensorQ mQ, TensorK mK, TensorV mV, TensorO mO,
    CUTLASS_GRID_CONSTANT const TmaQ tma_q,
    CUTLASS_GRID_CONSTANT const TmaK tma_k,
    CUTLASS_GRID_CONSTANT const TmaV tma_v,
    CUTLASS_GRID_CONSTANT const TmaO tma_o) {
    using SStorage = typename CONFIG_PARAMS::SStorage;
    constexpr int kBlockM  = CONFIG_PARAMS::kBlockM;
    constexpr int kBlockN  = CONFIG_PARAMS::kBlockN;
    constexpr int kHeadNum = CONFIG_PARAMS::kHeadNum;
    constexpr int kHeadDim = CONFIG_PARAMS::kHeadDim;
    constexpr int STAGE    = CONFIG_PARAMS::STAGE;
    constexpr int kProducerRegCount = 40;
    constexpr int kConsumerRegCount = 232;
    const int M_num = CEILDIV(seq_len, kBlockM);
    const int N_num = CEILDIV(seq_len, kBlockN);
    const int tiles_num = M_num*batch*kHeadNum;  
    constexpr auto cta_Q = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
    constexpr auto cta_K = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
    constexpr auto cta_V = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
    constexpr auto cta_O = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});

    constexpr int sQ_bytes = size(cta_Q) * sizeof(T);
    constexpr int sK_bytes = size(cta_K) * sizeof(T);
    constexpr int sV_bytes = size(cta_V) * sizeof(T);

    static_assert(STAGE == 2);
    extern __shared__ uint8_t smem[];
    SStorage& sstorage = *reinterpret_cast<SStorage*>(smem);

    if(threadIdx.y==0) {
        sstorage.p_qk.template init_barrier<1, (kNWarps<<5)>();
        sstorage.p_v.template init_barrier<1, (kNWarps<<5)>();
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();

    auto sK = sstorage.get_sK();
    auto sV = sstorage.get_sV();
    if(producer<4>()) {
        dealloc_warpgroup_registers<kProducerRegCount>();
        Pipeline<STAGE, sK_bytes> pipeline_qk(&(sstorage.p_qk), true);
        SimplePipeline<sV_bytes> pipeline_v(&(sstorage.p_v), true);
        for(int tiled_id=blockIdx.x; tiled_id<tiles_num; tiled_id+=gridDim.x) {
            if(threadIdx.x==0 && threadIdx.y==0) {
                auto [q_block, head_id, batch_id] = get_tile_coord<kHeadNum>(M_num, tiled_id);
                Tensor sQ = sstorage.get_sQ(pipeline_qk.index());
                Q_g2s<sQ_bytes>(tma_q, mQ(_, _, head_id, batch_id), cta_Q,
                            make_coord(q_block, _0{}), sQ, pipeline_qk);
                Tensor gK = local_tile(mK(_, _, head_id, batch_id), cta_K,  make_coord(_, _0{}));
                Tensor gV = local_tile(mV(_, _, head_id, batch_id), cta_V,  make_coord(_, _0{}));
                auto [tma_gK, tma_sK] = tma_partition(tma_k, group_modes<0, 2>(sK), group_modes<0, 2>(gK));
                auto [tma_gV, tma_sV] = tma_partition(tma_v, group_modes<0, 2>(sV), group_modes<0, 2>(gV));

                const int k_blocks = is_causal ? CEILDIV((q_block + 1) * kBlockM, kBlockN) : N_num;
                for(int i=0; i<k_blocks; ++i) {
                    pipeline_qk.template copy_g2s<sK_bytes>(tma_k, tma_gK(_, i), tma_sK(_, pipeline_qk.index()));
                    pipeline_v.copy_g2s(tma_v, tma_gV(_, i), tma_sV);
                }
            }
            tile_buffer_sync<kNWarps>();
        }
    }
    else {
        alloc_warpgroup_registers<kConsumerRegCount>();
        Pipeline<STAGE, sK_bytes> pipeline_qk(&(sstorage.p_qk), false);
        SimplePipeline<sV_bytes> pipeline_v(&(sstorage.p_v), false);
        const int wid = threadIdx.y-4;
        const int tid  = (wid<<5) + threadIdx.x;

        typename CONFIG_PARAMS::TiledMMA_QK mma_qk;
        typename CONFIG_PARAMS::TiledMMA_PV mma_pv;
        ThrMMA thr_qk = mma_qk.get_thread_slice(tid);
        Tensor rQ = thr_qk.partition_fragment_A(sstorage.get_sQ(_0{}));

        Tensor rS   = partition_fragment_C(mma_qk, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));        
        Tensor rS_RC= get_rowcol(rS);
        Tensor rO    = partition_fragment_C(mma_pv, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
        Tensor rO_RC = get_rowcol(rO);
        Tensor rS_max = make_tensor<float>(Shape<Int<2*size<1>(rS)>>{});
        Tensor rS_sum = make_fragment_like(rS_max);

        Copy_Atom<SM75_U32x4_LDSM_N, T> s2r_q;
        Copy_Atom<SM75_U32x4_LDSM_N, T> s2r_k;
        Copy_Atom<SM75_U16x8_LDSM_T, T> s2r_v;
        Copy_Atom<SM90_U32x4_STSM_N, T> r2s_o;

        for(int tiled_id=blockIdx.x; tiled_id<tiles_num; tiled_id+=gridDim.x) {
            clear(rO);
            clear(rS_sum);
            CUTE_UNROLL
            for(int m=0; m<size(rS_max); ++m)
                rS_max(m) = -INFINITY;
            auto [q_block, head_id, batch_id] = get_tile_coord<kHeadNum>(M_num, tiled_id);

            const int k_blocks = is_causal ? CEILDIV((q_block + 1) * kBlockM, kBlockN) : N_num;
            Q_s2r(sstorage, pipeline_qk, s2r_q, mma_qk, rQ, tid);

            for(int i=0; i<k_blocks; ++i) {
                clear(rS);
                gemm_flash<T, decltype(sK(_, _, _0{}))>(s2r_k, sK, tid,
                    rS, rQ, mma_qk, pipeline_qk);
                if constexpr(is_causal) {
                    if((i+1)*kBlockN > q_block*kBlockM)
                        causal_mask<kBlockN, kBlockM, kNWarps>(rS_RC, q_block, i);
                }
                softmax_rescale<T>(rS_RC, rS_max, rS_sum, rO_RC, scale);

                auto rP_h = convert_f2h<T>(rS);
                auto rP = make_tensor(rP_h.data(), Convert_Areg(rS.layout()));
                gemm_flash_v<T, decltype(sstorage.get_sVt_D())>(sstorage, s2r_v, tid, rO, rP, mma_pv, pipeline_v);
            }
            

            normalize_stats<kNWarps>(rS_max, rS_sum, rO_RC, scale);

            auto sO = sstorage.get_sO();
            auto rO_f16 = convert_f2h<T>(rO);
            O_r2s(r2s_o, mma_pv, rO_f16, sO, tid);
            bar_sync<kNWarps>();

            if(threadIdx.x == 0 && wid == 0) {
                O_s2g(tma_o, mO, cta_O, sO, q_block, head_id, batch_id);
                tma_store_wait<0>();
            }
            tile_buffer_sync<kNWarps>();
        }
    }
}

template<typename T, bool is_causal,
         int kHeadNum, int kHeadDim,
         int kBlockM, int kBlockN, int kNWarps, int STAGE>
void flash_fwd(    
    float scale,
    int seq_len,
    int batch,
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    cudaStream_t stream){
    using CONFIG_PARAMS = Flash_fwd_kernel_traits<
        kHeadNum, kHeadDim, 
        kBlockM, kBlockN, kNWarps, STAGE, T>; 

    CONFIG_PARAMS params;
    using SS = typename CONFIG_PARAMS::SStorage;
    auto[tma_q, mQ] = params.template get_tma<typename SS::LayoutQ>(Q, seq_len, batch);
    auto[tma_k, mK] = params.template get_tma<typename SS::LayoutKStage>(K, seq_len, batch);
    auto[tma_v, mV] = params.template get_tma<typename SS::LayoutKStage>(V, seq_len, batch);
    auto[tma_o, mO] = params.template get_tma<typename SS::LayoutO>(O, seq_len, batch, SM90_TMA_STORE{});
    auto kernel_func = &flash_atten_v3_fwd<
        T, is_causal, kNWarps, CONFIG_PARAMS,
        decltype(mQ), decltype(mK), decltype(mV), decltype(mO),
        decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_o)>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SS)));
    int sm_num=0;
    CUTE_CHECK_ERROR(
        cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, 0));
    
    int block_num = std::min(sm_num, CEILDIV(seq_len, kBlockM)*batch*kHeadNum);
    dim3 grid(block_num, 1, 1);
    dim3 block(32, 4+kNWarps, 1);
    kernel_func<<<grid, block, sizeof(SS), stream>>>(scale*M_LOG2E, seq_len, batch, mQ, mK, mV, mO, tma_q, tma_k, tma_v, tma_o);
}
