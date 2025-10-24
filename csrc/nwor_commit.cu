/*
 * NWOR Draft Commit Kernel Implementation
 *
 * This kernel scatters accepted tokens from staged draft buffers to the KV cache.
 * It reuses the exact vectorization and quantization logic from reshape_and_cache_flash_kernel
 * to ensure correctness and performance.
 */

#include "nwor_commit.h"

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#include "copy_with_scale_op.cuh"

namespace vllm {

// ============================================================================
// NWOR Kernels: Copy-on-Write Logging and Restoration
// ============================================================================

// Log cache slots to buffer (Copy-on-Write preparation)
// Runs during CUDA graph capture AND replay to keep log buffers fresh
template <typename scalar_t, typename cache_t>
__global__ void log_cache_slots_kernel(
    const cache_t* __restrict__ key_cache,
    const cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_indices,  // [num_slots] - cache slots to log
    scalar_t* __restrict__ log_key,            // [num_slots, num_heads, head_size]
    scalar_t* __restrict__ log_value,
    const int64_t block_size,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int num_heads,
    const int head_size
) {
    const int64_t log_idx = blockIdx.x;  // Which slot are we logging
    const int64_t slot_idx = slot_indices[log_idx];

    // Skip padding slots
    if (slot_idx < 0) return;

    // Decompose slot index into block coordinates
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n_elems = num_heads * head_size;

    // Source: cache at this slot
    const cache_t* __restrict__ key_src =
        key_cache + block_idx * block_stride + block_offset * page_stride;
    const cache_t* __restrict__ value_src =
        value_cache + block_idx * block_stride + block_offset * page_stride;

    // Destination: log buffer
    scalar_t* __restrict__ key_dst = log_key + log_idx * n_elems;
    scalar_t* __restrict__ value_dst = log_value + log_idx * n_elems;

    // Layout detection (same as restore kernel)
    const bool is_contiguous_heads = (head_stride == head_size);

    if (is_contiguous_heads) {
        // NHD layout - simple vectorized copy
        for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
            key_dst[i] = static_cast<scalar_t>(key_src[i]);
            value_dst[i] = static_cast<scalar_t>(value_src[i]);
        }
    } else {
        // HND layout - strided heads
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int warps_per_block = blockDim.x >> 5;

        for (int head = warp_id; head < num_heads; head += warps_per_block) {
            const cache_t* __restrict__ k_src_h =
                key_src + static_cast<int64_t>(head) * head_stride;
            const cache_t* __restrict__ v_src_h =
                value_src + static_cast<int64_t>(head) * head_stride;

            scalar_t* __restrict__ k_dst_h = key_dst + head * head_size;
            scalar_t* __restrict__ v_dst_h = value_dst + head * head_size;

            for (int i = lane; i < head_size; i += 32) {
                k_dst_h[i] = static_cast<scalar_t>(k_src_h[i]);
                v_dst_h[i] = static_cast<scalar_t>(v_src_h[i]);
            }
        }
    }
}

// Dispatch macro for log_cache_slots
#define CALL_LOG_CACHE_SLOTS_KERNEL(CACHE_T, SCALAR_T)  \
    log_cache_slots_kernel<SCALAR_T, CACHE_T>           \
        <<<grid, block, 0, stream>>>(                    \
            reinterpret_cast<const CACHE_T*>(key_cache.data_ptr()), \
            reinterpret_cast<const CACHE_T*>(value_cache.data_ptr()), \
            slot_indices.data_ptr<int64_t>(),            \
            reinterpret_cast<SCALAR_T*>(log_key.data_ptr()), \
            reinterpret_cast<SCALAR_T*>(log_value.data_ptr()), \
            block_size, block_stride, page_stride, head_stride, \
            static_cast<int>(num_heads), static_cast<int>(head_size))

void log_cache_slots(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_indices,
    torch::Tensor& log_key,
    torch::Tensor& log_value,
    int64_t block_size,
    int64_t block_stride,
    int64_t page_stride,
    int64_t head_stride
) {
    int num_slots = slot_indices.size(0);
    if (num_slots == 0) return;

    int64_t num_heads = log_key.size(1);
    int64_t head_size = log_key.size(2);

    dim3 grid(static_cast<unsigned int>(num_slots));
    dim3 block(static_cast<unsigned int>(std::min(num_heads * head_size, static_cast<int64_t>(512))));

    const at::cuda::OptionalCUDAGuard device_guard(key_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on cache dtype and log buffer dtype
    AT_DISPATCH_SWITCH(
        key_cache.scalar_type(), "log_cache_slots_cache",
        AT_DISPATCH_CASE(at::ScalarType::Byte, [&] {
            using cache_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                log_key.scalar_type(), "log_cache_slots_log", [&] {
                    CALL_LOG_CACHE_SLOTS_KERNEL(cache_t, scalar_t);
                });
        })
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            using cache_t = float;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                log_key.scalar_type(), "log_cache_slots_log", [&] {
                    CALL_LOG_CACHE_SLOTS_KERNEL(cache_t, scalar_t);
                });
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            using cache_t = __half;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                log_key.scalar_type(), "log_cache_slots_log", [&] {
                    CALL_LOG_CACHE_SLOTS_KERNEL(cache_t, scalar_t);
                });
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            using cache_t = __nv_bfloat16;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                log_key.scalar_type(), "log_cache_slots_log", [&] {
                    CALL_LOG_CACHE_SLOTS_KERNEL(cache_t, scalar_t);
                });
        })
    );
}

// Copy-on-write: Restore rejected draft slots from log buffers
// Similar to commit_draft_kernel but sources from log instead of staged tensors
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void restore_rejected_drafts_kernel(
    const scalar_t* __restrict__ log_key,    // [num_rejected, num_heads, head_size]
    const scalar_t* __restrict__ log_value,  // [num_rejected, num_heads, head_size]
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const int32_t* __restrict__ slot_indices,  // [num_rejected] - cache slots to restore
    const float* log_k_scale,  // [num_rejected] or empty
    const float* log_v_scale,  // [num_rejected] or empty
    const bool scale_is_per_token,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int num_heads,
    const int head_size,
    const int block_size
) {
    const int64_t rejected_idx = blockIdx.x;
    const int64_t slot_idx = slot_indices[rejected_idx];

    // Skip padding slots
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n_elems = num_heads * head_size;

    // Source: log buffers
    const scalar_t* __restrict__ key_src = log_key + rejected_idx * num_heads * head_size;
    const scalar_t* __restrict__ value_src = log_value + rejected_idx * num_heads * head_size;

    // Destination: cache at the slot we're restoring
    cache_t* __restrict__ key_dst =
        key_cache + block_idx * block_stride + block_offset * page_stride;
    cache_t* __restrict__ value_dst =
        value_cache + block_idx * block_stride + block_offset * page_stride;

    // Layout detection
    const bool is_contiguous_heads = (head_stride == head_size);

    // Quantization scales (FIX BUG #1: Handle both per-token and scalar scales)
    float k_scale_val = 0.f;
    float v_scale_val = 0.f;
    if constexpr (kv_dt != Fp8KVCacheDataType::kAuto) {
        if (log_k_scale != nullptr) {
            k_scale_val = scale_is_per_token ? log_k_scale[rejected_idx] : log_k_scale[0];
        }
        if (log_v_scale != nullptr) {
            v_scale_val = scale_is_per_token ? log_v_scale[rejected_idx] : log_v_scale[0];
        }
    }

    // Vectorized copy with quantization
    constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    if (is_contiguous_heads) {
        // NHD layout (FIX BUG #4: Remove unnecessary namespace prefix)
        vectorize_with_alignment<VEC_SIZE>(
            key_src, key_dst, n_elems, threadIdx.x, blockDim.x, k_op);
        vectorize_with_alignment<VEC_SIZE>(
            value_src, value_dst, n_elems, threadIdx.x, blockDim.x, v_op);
    } else {
        // HND layout (strided heads)
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int warps_per_block = blockDim.x >> 5;

        for (int head = warp_id; head < num_heads; head += warps_per_block) {
            const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
            const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

            cache_t* __restrict__ k_dst_h =
                key_dst + static_cast<int64_t>(head) * head_stride;
            cache_t* __restrict__ v_dst_h =
                value_dst + static_cast<int64_t>(head) * head_stride;

            vectorize_with_alignment<VEC_SIZE>(
                k_src_h, k_dst_h, head_size, lane, 32, k_op);
            vectorize_with_alignment<VEC_SIZE>(
                v_src_h, v_dst_h, head_size, lane, 32, v_op);
        }
    }
}

// Template dispatch macro (inside namespace, no prefix needed)
#define CALL_RESTORE_REJECTED_KERNEL(KV_T, CACHE_T, KV_DTYPE)             \
    restore_rejected_drafts_kernel<KV_T, CACHE_T, KV_DTYPE>               \
        <<<grid, block, 0, stream>>>(                                      \
            reinterpret_cast<const KV_T*>(log_key.data_ptr()),             \
            reinterpret_cast<const KV_T*>(log_value.data_ptr()),           \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),              \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),            \
            slot_indices.data_ptr<int32_t>(),                              \
            log_k_scale.numel() > 0 ?                                      \
                reinterpret_cast<const float*>(log_k_scale.data_ptr()) : nullptr, \
            log_v_scale.numel() > 0 ?                                      \
                reinterpret_cast<const float*>(log_v_scale.data_ptr()) : nullptr, \
            scale_is_per_token,                                            \
            block_stride, page_stride, head_stride,                        \
            static_cast<int>(num_heads), static_cast<int>(head_size),     \
            static_cast<int>(block_size))

void restore_rejected_drafts(
    torch::Tensor& log_key,
    torch::Tensor& log_value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_indices,
    int64_t block_size,
    int64_t block_stride,
    int64_t page_stride,
    int64_t head_stride,
    const std::string& kv_cache_dtype,
    torch::Tensor& log_k_scale,
    torch::Tensor& log_v_scale
) {
    // Validate inputs
    TORCH_CHECK(slot_indices.dtype() == torch::kInt32,
                "slot_indices must have dtype torch.int32");
    TORCH_CHECK(log_key.device() == log_value.device(),
                "log_key and log_value must be on the same device");
    TORCH_CHECK(log_key.device() == key_cache.device(),
                "log_key and key_cache must be on the same device");

    int num_rejected = slot_indices.size(0);
    if (num_rejected == 0) {
        return;  // Nothing to restore
    }

    int64_t num_heads = log_key.size(1);
    int64_t head_size = log_key.size(2);

    // Determine if scales are per-token (FIX BUG #2: > 1 not > 0)
    // numel == 0: no scales, numel == 1: scalar scale, numel > 1: per-token scales
    bool scale_is_per_token = (log_k_scale.numel() > 1);

    // Grid/block dimensions
    dim3 grid(static_cast<unsigned int>(num_rejected));
    dim3 block(static_cast<unsigned int>(std::min(num_heads * head_size, static_cast<int64_t>(512))));

    // Device guard and stream
    const at::cuda::OptionalCUDAGuard device_guard(log_key.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch kernel
    DISPATCH_BY_KV_CACHE_DTYPE(log_key.dtype(), kv_cache_dtype, CALL_RESTORE_REJECTED_KERNEL);
}

}  // namespace vllm
