#pragma once
// #include "cute/algorithm/tensor_reduce.hpp"
#include "utils/tools.h"

namespace flash {

template <class SrcEngine, class SrcLayout, class T, class BinaryOp = cute::plus>
CUTE_HOST_DEVICE constexpr
T
reduce(Tensor<SrcEngine,SrcLayout> const& src, T init, BinaryOp op = {})
{
  #pragma unroll
  for (auto i = 0; i < size(src); ++i) {
    init = op(init, src(i));
  }
  return init;
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout,
          class BinaryOp = cute::plus>
CUTE_HOST_DEVICE constexpr
void
batch_reduce(Tensor<SrcEngine, SrcLayout> const& src,       // (BatchMode, RedMode)
             Tensor<DstEngine, DstLayout>      & dst,       // (BatchMode)
             BinaryOp op = {})
{
  // Precondition
  CUTE_STATIC_ASSERT_V(rank(src) == Int<2>{});
  assert(size<0>(src) == size(dst));
  
  #pragma unroll
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = reduce(src(i,_), dst(i), op);
  }
}

template<int Num>
__forceinline__ __device__ 
auto warp_reduce_max(auto data) {
    #pragma unroll
    for (uint32_t mask = Num/2; mask > 0; mask >>= 1) {
        data = fmaxf(data, __shfl_xor_sync(0xffffffff, data, mask));
    }
    return data;
}

template<int Num>
__forceinline__ __device__ 
auto warp_reduce_sum(auto data) {
    #pragma unroll
    for (uint32_t mask = Num/2; mask > 0; mask >>= 1) {
        data +=  __shfl_xor_sync(0xffffffff, data, mask);
    }
    return data;
}

template<int Num>
__forceinline__ __device__ 
void reduce_max(auto& score, auto& score_max) {
    flash::batch_reduce(score, score_max, cute::max_fn{});
    #pragma unroll
    for(int i=0; i<size(score_max); ++i) {
        score_max(i) = warp_reduce_max<Num>(score_max(i));
    }
}

}
