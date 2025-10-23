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
// NWOR Kernels: commit_draft and restore_rejected_drafts
// ============================================================================

// Commit draft kernel - copied vectorization from reshape_and_cache_flash_kernel
// Key differences:
// 1. Early exit on mask[token_idx] == false (Issue #3: mask early-return)
// 2. No atomic count (Issue #5: count computed in Python)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void commit_draft_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // Layout-dependent, see head_stride check
    cache_t* __restrict__ value_cache,
    const bool* __restrict__ mask,             // [num_tokens]
    const int32_t* __restrict__ slot_mapping,  // [num_tokens] - guaranteed int32
    const float* k_scale,
    const float* v_scale,
    const bool scale_is_per_token,
    const int64_t key_stride,
    const int64_t value_stride,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int num_heads,
    const int head_size,
    const int block_size
) {
    const int64_t token_idx = blockIdx.x;

    // Issue #3: Mask early-return BEFORE any other work (avoid divergence)
    if (!mask[token_idx]) {
        return;
    }

    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n_elems = num_heads * head_size;

    // Pointers to the beginning of the source row for this token
    const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
    const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

    // Find the start position inside the kv-cache for this token
    cache_t* __restrict__ key_dst =
        key_cache + block_idx * block_stride + block_offset * page_stride;
    cache_t* __restrict__ value_dst =
        value_cache + block_idx * block_stride + block_offset * page_stride;

    // This is true for the NHD layout where `head_stride == head_size`
    const bool is_contiguous_heads = (head_stride == head_size);

    // Issue #3: Quantization scale handling (per-token vs scalar)
    float k_scale_val = 0.f;
    float v_scale_val = 0.f;
    if constexpr (kv_dt != Fp8KVCacheDataType::kAuto) {
        if (k_scale != nullptr) {
            k_scale_val = scale_is_per_token ? k_scale[token_idx] : k_scale[0];
        }
        if (v_scale != nullptr) {
            v_scale_val = scale_is_per_token ? v_scale[token_idx] : v_scale[0];
        }
    }

    // Issue #4: Exact vectorization copied from reshape_and_cache_flash_kernel
    constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    if (is_contiguous_heads) {
        // NHD layout: [num_blocks, block_size, num_heads, head_size]
        vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                           blockDim.x, k_op);

        vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                           threadIdx.x, blockDim.x, v_op);

    } else {
        // HND layout: [num_blocks, num_heads, block_size, head_size]
        // Heads are strided, but each head_size segment is contiguous
        const int lane = threadIdx.x & 31;     // 0..31 within warp
        const int warp_id = threadIdx.x >> 5;  // warp index within block
        const int warps_per_block = blockDim.x >> 5;

        for (int head = warp_id; head < num_heads; head += warps_per_block) {
            const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
            const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

            cache_t* __restrict__ k_dst_h =
                key_dst + static_cast<int64_t>(head) * head_stride;
            cache_t* __restrict__ v_dst_h =
                value_dst + static_cast<int64_t>(head) * head_stride;

            // Within each head, let the 32 threads of the warp perform the vector copy
            vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                               k_op);

            vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                               v_op);
        }
    }
}

// Template dispatch macro (inside namespace, no vllm:: prefix needed)
#define CALL_COMMIT_DRAFT_KERNEL(KV_T, CACHE_T, KV_DTYPE)               \
  commit_draft_kernel<KV_T, CACHE_T, KV_DTYPE>                          \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<const KV_T*>(key_ptr),                       \
          reinterpret_cast<const KV_T*>(value_ptr),                     \
          reinterpret_cast<CACHE_T*>(key_cache_ptr),                    \
          reinterpret_cast<CACHE_T*>(value_cache_ptr),                  \
          reinterpret_cast<const bool*>(mask_ptr),                      \
          reinterpret_cast<const int32_t*>(slot_ptr),                   \
          k_scale_ptr ? reinterpret_cast<const float*>(k_scale_ptr) : nullptr, \
          v_scale_ptr ? reinterpret_cast<const float*>(v_scale_ptr) : nullptr, \
          scale_is_per_token,                                           \
          key_stride, value_stride, block_stride, page_stride,          \
          head_stride, static_cast<int>(num_heads), static_cast<int>(head_size), static_cast<int>(block_size))

// ============================================================================
// Wrapper Functions
// ============================================================================

void commit_draft_layer(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& mask,
    torch::Tensor& slot_mapping,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::string& kv_cache_dtype
) {
    // Validate dtypes
    TORCH_CHECK(mask.dtype() == torch::kBool,
                "mask must have dtype torch.bool, got ", mask.dtype());
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt32,
                "slot_mapping must have dtype torch.int32, got ", slot_mapping.dtype());

    // Validate devices
    TORCH_CHECK(key.device() == value.device(),
                "key and value must be on the same device");
    TORCH_CHECK(key.device() == key_cache.device(),
                "key and key_cache must be on the same device");
    TORCH_CHECK(key.device() == value_cache.device(),
                "key and value_cache must be on the same device");
    TORCH_CHECK(key.device() == mask.device(),
                "key and mask must be on the same device");
    TORCH_CHECK(key.device() == slot_mapping.device(),
                "key and slot_mapping must be on the same device");

    // Validate strides
    TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0),
                "key_cache and value_cache must have the same block stride");

    // Extract tensor dimensions
    // key/value: [num_tokens, num_heads, head_size]
    int num_tokens = slot_mapping.size(0);
    int64_t num_heads = key.size(1);
    int64_t head_size = key.size(2);

    // Validate tensor sizes
    TORCH_CHECK(mask.numel() >= num_tokens,
                "mask size (", mask.numel(), ") must be >= num_tokens (", num_tokens, ")");
    if (k_scale.numel() > 0) {
        TORCH_CHECK(k_scale.device() == key.device(),
                    "k_scale must be on the same device as key");
    }
    if (v_scale.numel() > 0) {
        TORCH_CHECK(v_scale.device() == key.device(),
                    "v_scale must be on the same device as key");
    }

    // FIX BUG #3: Use stride-based layout detection (robust when block_size == num_heads)
    // key_cache layouts:
    //   Flash: [num_blocks, block_size, num_heads, head_size] - stride(1) > stride(2)
    //   Paged: [num_blocks, num_heads, block_size, head_size] - stride(2) > stride(1)
    // Shape-based detection (size(1) == num_heads) FAILS when block_size == num_heads!

    // Compute strides first (matching cache_kernels.cu:715-716)
    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);
    int64_t block_stride = key_cache.stride(0);
    int64_t page_stride = key_cache.stride(1);
    int64_t head_stride = key_cache.stride(2);

    // Detect layout via stride comparison (when strides differ, larger one is the page dim)
    int64_t block_size;
    if (key_cache.stride(1) != key_cache.stride(2)) {
        // Flash: stride(1) larger (page dimension)
        // Paged: stride(2) larger (page dimension)
        block_size = (key_cache.stride(1) > key_cache.stride(2)) ?
                     key_cache.size(1) : key_cache.size(2);
    } else {
        // Strides equal (unusual), fall back to size comparison
        block_size = key_cache.size(1);
    }
    // Kernel auto-detects NHD vs HND layout via: (head_stride == head_size)

    // Determine if scales are per-token
    bool scale_is_per_token = (k_scale.numel() > 1);

    // Extract pointers
    int64_t key_ptr = reinterpret_cast<int64_t>(key.data_ptr());
    int64_t value_ptr = reinterpret_cast<int64_t>(value.data_ptr());
    int64_t key_cache_ptr = reinterpret_cast<int64_t>(key_cache.data_ptr());
    int64_t value_cache_ptr = reinterpret_cast<int64_t>(value_cache.data_ptr());
    int64_t mask_ptr = reinterpret_cast<int64_t>(mask.data_ptr());
    int64_t slot_ptr = reinterpret_cast<int64_t>(slot_mapping.data_ptr());
    int64_t k_scale_ptr = k_scale.numel() > 0 ? reinterpret_cast<int64_t>(k_scale.data_ptr()) : 0;
    int64_t v_scale_ptr = v_scale.numel() > 0 ? reinterpret_cast<int64_t>(v_scale.data_ptr()) : 0;

    // Grid/block dimensions
    dim3 grid(static_cast<unsigned int>(num_tokens));
    dim3 block(static_cast<unsigned int>(std::min(num_heads * head_size, static_cast<int64_t>(512))));

    // Device guard and stream
    const at::cuda::OptionalCUDAGuard device_guard(key.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch kernel
    DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, CALL_COMMIT_DRAFT_KERNEL);
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
            reinterpret_cast<KV_T*>(log_key.data_ptr()),                   \
            reinterpret_cast<KV_T*>(log_value.data_ptr()),                 \
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
