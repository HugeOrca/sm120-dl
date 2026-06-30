#pragma once
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>

using namespace cute;

using bf16 = cutlass::bfloat16_t;
using bf162 = __nv_bfloat162;

#define CEILDIV(x, align) (x+align-1)/align
#define ALIGNUP(x, align) (((x+align-1)/align)*align)

#define dv_flag  __device__ __forceinline__

template<int NProducerWarps = 1>
dv_flag bool producer() {
    return (threadIdx.y < NProducerWarps);
}

template<int NWarps, int producer_reg>
dv_flag constexpr int get_consumer_registers() {
   constexpr int raw = ((64*1024 - 128*producer_reg)/(NWarps*32))/8 * 8;
   return raw > 256 ? 256 : raw;
}

template<int BS_W> dv_flag
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
