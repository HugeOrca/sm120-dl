#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include "kernel_traits.hpp"

#define dv_flag  __device__ __forceinline__


template <
    int STAGE,
    uint32_t TransactionBytes = 0>
struct Pipeline {
private:
    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    using State = cutlass::PipelineState<STAGE>;

public:
    static_assert(STAGE > 0, "STAGE must be positive");
    PipelineStorage<STAGE>* storage = nullptr;
    State state;
    dv_flag Pipeline() = delete;
    dv_flag Pipeline(PipelineStorage<STAGE>* storage_, bool producer = false)
        : storage(storage_)
        , state(producer ? State{0, 1, 0} : State{}) {}

    template <uint32_t Bytes = TransactionBytes>
    dv_flag void acquire() {
        static_assert(Bytes > 0, "TransactionBytes must be non-zero");
        EmptyBarrier::wait(&storage->empty_barrier[state.index()], state.phase());
        FullBarrier::arrive_and_expect_tx(&storage->full_barrier[state.index()], Bytes);   
    }

    template <uint32_t Bytes = TransactionBytes, class Tma, class SrcTensor, class DstTensor>
    dv_flag void copy_g2s(Tma const& tma, SrcTensor const& src_tensor, DstTensor&& dst_tensor) {
        acquire<Bytes>();
        ::cute::copy(tma.with(storage->full_barrier[state.index()]), src_tensor, static_cast<DstTensor&&>(dst_tensor));
        ++(*this);
    }

    dv_flag void wait() {
        FullBarrier::wait(&storage->full_barrier[state.index()], state.phase());
    }

    dv_flag void release() {
        EmptyBarrier::arrive(&storage->empty_barrier[state.index()]);
        ++(*this);
    }

    dv_flag void operator++() {
        ++state;
    }
    
    dv_flag uint32_t index() const {
        return state.index();
    }
};

template <
    int STAGE,
    uint32_t TransactionBytes>
struct SimplePipeline {
private:
    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    using State = cutlass::PipelineState<STAGE>;

public:
    static_assert(STAGE > 0, "STAGE must be positive");
    static_assert(TransactionBytes > 0, "TransactionBytes must be non-zero");
    PipelineStorage<STAGE>* storage = nullptr;
    State state;

    dv_flag SimplePipeline() = delete;
    dv_flag SimplePipeline(PipelineStorage<STAGE>* storage_, bool producer = false)
        : storage(storage_)
        , state(producer ? State{0, 1, 0} : State{}) {}

    template <class Tma, class SrcTensor, class DstTensor>
    dv_flag void copy_g2s(Tma const& tma, SrcTensor const& src_tensor, DstTensor&& dst_tensor) {
        EmptyBarrier::wait(&storage->empty_barrier[state.index()], state.phase());
        FullBarrier::arrive_and_expect_tx(&storage->full_barrier[state.index()], TransactionBytes);
        ::cute::copy(tma.with(storage->full_barrier[state.index()]), src_tensor, static_cast<DstTensor&&>(dst_tensor));
        ++state;
    }

    dv_flag void wait() {
        FullBarrier::wait(&storage->full_barrier[state.index()], state.phase());
    }

    dv_flag void release() {
        EmptyBarrier::arrive(&storage->empty_barrier[state.index()]);
        ++state;
    }
};
