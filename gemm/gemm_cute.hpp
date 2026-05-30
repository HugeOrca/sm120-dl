#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_conversion.h"

#include "utils/tools.h"

using bf16 = cutlass::bfloat16_t;
using bf162 = __nv_bfloat162;

using namespace cute;
using namespace GMMA;

#define dv_flag  __device__ __forceinline__

template<typename T, class LayoutA, class LayoutB, class LayoutC>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<T, cosize_v<LayoutA>> A;
    alignas(128) cute::ArrayEngine<T, cosize_v<LayoutB>> B;
    alignas(128) cute::ArrayEngine<T, cosize_v<LayoutC>> C;
    static_assert(size<2>(LayoutA{}) == size<2>(LayoutB{}),
                  "SharedStorage requires A and B layouts to have the same pipeline stage count.");
    constexpr static int STAGE = size<2>(LayoutA{});
    uint64_t full_barrier[STAGE];
    uint64_t empty_barrier[STAGE];

    dv_flag constexpr auto get_sA(){
        return make_tensor(make_smem_ptr(A.begin()), LayoutA{});
    }
    dv_flag constexpr auto get_sB(){
        return make_tensor(make_smem_ptr(B.begin()), LayoutB{});
    }
    dv_flag constexpr auto get_sC(){
        return make_tensor(make_smem_ptr(C.begin()), LayoutC{});
    }
};


dv_flag bool producer() {
    return (threadIdx.y < 4);
}

template<int BS_W>
dv_flag
void block_swizzle(int tiled_id, int& bm, int& bn, int BM, int BN) {
    int SW = BS_W < BN ? BS_W : BN;
    int square = SW*BM;
    int num    = tiled_id / square;
    int base_col = num*SW;
    int SW2 = (base_col+SW) < BN ? SW : BN-base_col;
    int in_square = tiled_id%(SW2*BM);
    bm = in_square / SW2;
    bn = base_col + in_square % SW2;
}

/*grid:(SMs) block:(32, CWG*4+4)*/
template <typename T, typename SStorage, int BM, int BN, int BK, int STAGE, int CWG,
         int WARPS_M, int WARPS_N, int BS_W,
         class Shape, class TensorA, class TensorB, class TensorC, class MMA,
         class TmaA, class TmaB, class TmaC,
         int WARP_SIZE=32>
__global__ void
__launch_bounds__((CWG*4+4)*WARP_SIZE, 1)
    b16_gemm(
    Shape shape,
    TensorA mA, TensorB mB, TensorC mC,
    CUTLASS_GRID_CONSTANT const TmaA tma_a,
    CUTLASS_GRID_CONSTANT const TmaB tma_b,
    CUTLASS_GRID_CONSTANT const TmaC tma_c,
    float alpha, float beta) {
#if !defined(CUTE_ARCH_TMA_SM90_ENABLED)
    static_assert(sizeof(T) == 0,
                  "gemm_cute.hpp uses TMA and must be compiled for a feature-suffixed architecture such as sm_120a.");
#endif
    static_assert(WARP_SIZE == 32, "WARP_SIZE must be 32.");
    static_assert(STAGE > 0, "STAGE must be positive.");
    static_assert(CWG > 0, "CWG must be positive.");
    static_assert(SStorage::STAGE == STAGE, "Kernel STAGE must match SharedStorage stage count.");
    static_assert((BM % 16 == 0) && (BN % 8 == 0) && (BK % 16 == 0),
                  "BM/BN/BK must be compatible with the 16x8x16 MMA atom.");
    static_assert((CWG*4+4)*WARP_SIZE <= 1024, "Thread block size must not exceed 1024 threads.");
    static_assert(STAGE<=3);
    CUTE_STATIC_ASSERT_V(rank(shape)     == _3{});
    auto [M, N, K]  = shape;
    int M_NUM = M/BM;
    int N_NUM = N/BN;
    int K_NUM = K/BK;
    int tiles_num = M_NUM * N_NUM;
    constexpr int Consumer = CWG<<2;
    constexpr auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    extern __shared__ uint8_t smem_array[];
    SStorage& sstorage = *reinterpret_cast<SStorage*>(smem_array);
    Tensor sA = sstorage.get_sA();  // (BM, BK, STAGE)
    Tensor sB = sstorage.get_sB();  // (BM, BK, STAGE)
    Tensor sC = sstorage.get_sC();  // (BM, BN)
    auto& full_bar  = sstorage.full_barrier;
    auto& empty_bar = sstorage.empty_barrier;
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;
    if(threadIdx.y==0) {
        if(threadIdx.x < STAGE) {
            ProducerBar::init(&full_bar[threadIdx.x], 1);
            ConsumerBar::init(&empty_bar[threadIdx.x], Consumer<<5);
        }
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();

    auto write_state = cutlass::PipelineState<STAGE>();
    auto read_state  = cutlass::PipelineState<STAGE>();

    if(producer()) {
        if(threadIdx.y==0 && threadIdx.x==0) {
            int total_k=0;
            for(int tiled_id=blockIdx.x; tiled_id<tiles_num; tiled_id += gridDim.x) {
                int bm, bn;
                block_swizzle<BS_W>(tiled_id, bm, bn, M_NUM, N_NUM);
                auto cta_coord = make_coord(bm, bn, _);
                Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
                Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1, _1>{});
                auto [tma_gA, tma_sA] = tma_partition(
                tma_a, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
                auto [tma_gB, tma_sB] = tma_partition(
                tma_b, group_modes<0, 2>(sB), group_modes<0, 2>(gB));

                constexpr int transfer_bytes = (BM*BK + BN*BK) * sizeof(T);
                for(int k=0; k<K_NUM; k++) {
                    int pipe = write_state.index();
                    if(total_k >= STAGE) {
                        ConsumerBar::wait(&empty_bar[pipe], write_state.phase()^1);
                    }
                    ProducerBar::arrive_and_expect_tx(&full_bar[pipe], transfer_bytes);
                    copy(tma_a.with(full_bar[pipe]), tma_gA(_, k), tma_sA(_, pipe));
                    copy(tma_b.with(full_bar[pipe]), tma_gB(_, k), tma_sB(_, pipe));
                    ++write_state;
                    ++total_k;
                }
            }
        }
    }
    else {
        Copy_Atom<SM75_U32x4_LDSM_N, T> s2r_atom_a;
        Copy_Atom<SM75_U32x2_LDSM_N, T> s2r_atom_b;
        Copy_Atom<SM90_U32x2_STSM_N, T> r2s_atom_c;
        auto bar_sync = [&] () {
            cutlass::arch::NamedBarrier::sync(Consumer<<5, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        };

        const int wid = threadIdx.y - 4;
        const int tid = wid*WARP_SIZE + threadIdx.x;
        MMA mma;
        ThrMMA thr_mma = mma.get_thread_slice(tid);
        Tensor rA = thr_mma.partition_fragment_A(sA(_, _, _0{}));  // rA:(MMA,MMA_M,MMA_K)
        Tensor rB = thr_mma.partition_fragment_B(sB(_, _, _0{}));  // rB:(MMA,MMA_N,MMA_K)
        Tensor rC = thr_mma.partition_fragment_C(sC);  // rC:(MMA,MMA_M,MMA_N)
        auto rC_b16 = make_fragment_like<T>(rC);

        TiledCopy s2r_a = make_tiled_copy_A(s2r_atom_a, mma);
        TiledCopy s2r_b = make_tiled_copy_B(s2r_atom_b, mma);
        TiledCopy r2s_c = make_tiled_copy_C(r2s_atom_c, mma);
        auto thr_s2r_a = s2r_a.get_slice(tid);
        auto thr_s2r_b = s2r_b.get_slice(tid);
        auto thr_r2s_c = r2s_c.get_slice(tid);

        Tensor cpy_sA = thr_s2r_a.partition_S(sA);  // (CPY,MMA_M,MMA_K,PIPE)
        Tensor cpy_sB = thr_s2r_b.partition_S(sB);  // (CPY,MMA_N,MMA_K,PIPE)
        Tensor cpy_rC = thr_r2s_c.retile_S(rC_b16); // (CPY, CPY_M, CPY_N)
        Tensor cpy_rA = thr_s2r_a.retile_D(rA);     // (CPY,MMA_M,MMA_K)
        Tensor cpy_rB = thr_s2r_b.retile_D(rB);     // (CPY,MMA_N,MMA_K)
        Tensor cpy_sC = thr_r2s_c.partition_D(sC);  // (CPY, CPY_M, CPY_N)
        constexpr int K_step = size<2>(rA);

        bool store_in_flight = false;
        cutlass::NumericArrayConverter<T, float, 2> cvt;
        using SRC = cutlass::Array<float, 2>;
        using DST = cutlass::Array<T, 2>;

        for(int tiled_id=blockIdx.x; tiled_id<tiles_num; tiled_id+=gridDim.x) {
            int bm, bn;
            block_swizzle<BS_W>(tiled_id, bm, bn, M_NUM, N_NUM);
            auto cta_coord = make_coord(bm, bn);
            Tensor gC = local_tile(mC, make_shape(Int<BM>{}, Int<BN>{}), cta_coord);
            auto [tma_gC, tma_sC] = tma_partition(tma_c, group_modes<0,2>(sC), group_modes<0,2>(gC));
            clear(rC);

            auto consume_pipe = [&] (auto pipe_const) {
                constexpr int pipe = decltype(pipe_const)::value;
                ProducerBar::wait(&full_bar[pipe], read_state.phase());

                CUTE_UNROLL
                for(int kk=0; kk<K_step; kk++) {
                    copy(s2r_atom_b, cpy_sB(_, _, kk, pipe_const), cpy_rB(_, _, kk));
                }

                CUTE_UNROLL
                for(int kk=0; kk<K_step; kk++) {
                    copy(s2r_atom_a, cpy_sA(_, _, kk, pipe_const), cpy_rA(_, _, kk));
                    gemm(mma, rA(_, _, kk), rB(_, _, kk), rC);
                }

                ConsumerBar::arrive(&empty_bar[pipe]);
            };

            for(int k=0; k<K_NUM; k++) {
                int read_pipe = read_state.index();
                switch (read_pipe) {
                    case 0: {
                        consume_pipe(Int<0>{});
                        break;
                    }
                    case 1: {
                        consume_pipe(Int<1>{});
                        break;
                    }
                    case 2: {
                        consume_pipe(Int<2>{});
                        break;
                    }
                    default: {
                        assert(false);
                    }
                }
                ++read_state;
            }

            auto rC_src = recast<SRC>(rC);
            auto rC_dst = recast<DST>(rC_b16);
            CUTE_UNROLL
            for(int i=0; i<rC_src.size(); i++) {
                rC_dst[i] = cvt(rC_src[i]);
            }

            if(store_in_flight && threadIdx.x == 0 && wid == 0) {
                tma_store_wait<0>();
            }
            bar_sync();

            copy(r2s_c, cpy_rC, cpy_sC);
            tma_store_fence();
            bar_sync();
            if(threadIdx.x == 0 && wid == 0) {
                copy(tma_c, tma_sC, tma_gC);
                tma_store_arrive();
            }
            store_in_flight = true;
        }

        if(store_in_flight && threadIdx.x == 0 && wid == 0) {
            tma_store_wait<0>();
        }
    }
    __syncthreads();
    if(threadIdx.y==0) {
        if(threadIdx.x < STAGE) {
            ProducerBar::invalidate(&full_bar[threadIdx.x]);
            ConsumerBar::invalidate(&empty_bar[threadIdx.x]);
        }
    }
}

template<typename T,
         int BM, int BN, int BK,
         int STAGE, int CWG,
         int WARPS_M, int WARPS_N, int BS_W>  // EU={WARPS_M, WARPS_N, 1}
void gemm_tn(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    cudaStream_t stream) {

    static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, cutlass::half_t>,
                  "gemm_tn supports only cutlass::bfloat16_t or cutlass::half_t.");
    static_assert(sizeof(T) == 2, "gemm_tn expects a 16-bit element type.");
    static_assert(BM > 0 && BN > 0 && BK > 0, "BM, BN, and BK must be positive.");
    static_assert(STAGE > 0, "STAGE must be positive.");
    static_assert(CWG > 0, "CWG must be positive.");
    static_assert(WARPS_M > 0 && WARPS_N > 0, "WARPS_M and WARPS_N must be positive.");
    static_assert(BS_W > 0, "BS_W must be positive.");

    if(M%BM!=0 || N%BN!=0 || K%BK!=0) {
        fprintf(stderr, "M, N, K must be divisible by BM, BN, BK.\n");
        exit(1);
    }

    auto global_shape = make_shape(M, N, K);

    static_assert(BM%32 ==0, "BM must be divisible by 32");
    static_assert(BK%64 ==0, "BK must be divisible by 64");
    static_assert(BN%64 ==0, "BN must be divisible by 64");
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<BM>{}, Int<BK>{}, Int<STAGE>{}));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<BN>{}, Int<BK>{}, Int<STAGE>{}));
    auto sC = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<BM>{}, Int<BN>{}));

    auto mA = make_tensor(A, make_shape(M, K),make_stride(K, _1{}));
    auto mB = make_tensor(B, make_shape(N, K),make_stride(K, _1{}));
    auto mC = make_tensor(C, make_shape(M, N),make_stride(N, _1{}));

    Copy_Atom tma_a = make_tma_atom(SM90_TMA_LOAD{},  mA, sA(_,_,Int<0>{}), make_shape(Int<BM>{}, Int<BK>{}));
    Copy_Atom tma_b = make_tma_atom(SM90_TMA_LOAD{},  mB, sB(_,_,Int<0>{}), make_shape(Int<BN>{}, Int<BK>{}));
    Copy_Atom tma_c = make_tma_atom(SM90_TMA_STORE{}, mC, sC,               make_shape(Int<BM>{}, Int<BN>{}));
    Tensor mA_tma = tma_a.get_tma_tensor(shape(mA));
    Tensor mB_tma = tma_b.get_tma_tensor(shape(mB));
    Tensor mC_tma = tma_c.get_tma_tensor(shape(mC));

    using mma_traits = std::conditional_t<
        std::is_same_v<T, bf16>,
        MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>,
        MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>;
    using mma_atom = MMA_Atom<mma_traits>;

    static_assert(WARPS_M*WARPS_N == 4*CWG, "The number of warps must be equal to 4*CWG.");
    static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0, "BM and BN must be divisible by WARPS_M and WARPS_N.");
    constexpr auto EU = make_layout(make_shape(Int<WARPS_M>{}, Int<WARPS_N>{}));
    // using MMA_Permute = Tile<Int<BM>, Int<BN>, _16>;
    constexpr int WM = BM/(16*WARPS_M);
    constexpr int WN = BN/(8*WARPS_N);
    using MMA_Permute = Tile<
    Layout<Shape<_16, Int<WARPS_M>, Int<WM>>, Stride<_1, Int<16*WM>, _16>>,
    Layout<Shape<_8, Int<WARPS_N>, Int<WN>>, Stride<_1, Int<8*WN>, _8>>,
    _16
    >;
    using MMA = decltype(make_tiled_mma(mma_atom{}, EU, MMA_Permute{}));

    using SStorage = SharedStorage<T, decltype(sA), decltype(sB), decltype(sC)>;
    size_t ssize = sizeof(SStorage);
    auto* kernel_func = &b16_gemm<
        T, SStorage, BM, BN, BK, STAGE, CWG, WARPS_M, WARPS_N, BS_W,
        decltype(global_shape), decltype(mA_tma), decltype(mB_tma), decltype(mC_tma), MMA,
        decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, ssize));

    int sm_num=0;
    CUTE_CHECK_ERROR(
        cudaDeviceGetAttribute(
            &sm_num, cudaDevAttrMultiProcessorCount, 0));

    int block_num = std::min(sm_num, (M/BM)*(N/BN));
    constexpr int WARP_SIZE = 32;
    dim3 block(WARP_SIZE, 4*CWG+4);
    dim3 grid(block_num);

    kernel_func<<<grid, block, ssize, stream>>>(
        global_shape, mA_tma, mB_tma, mC_tma, tma_a, tma_b, tma_c, 1, 0);
    CUTE_CHECK_LAST();
}
