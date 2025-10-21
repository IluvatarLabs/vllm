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

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

namespace vllm {

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

}  // namespace vllm

// Template dispatch macro matching existing pattern
#define CALL_COMMIT_DRAFT_KERNEL(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::commit_draft_kernel<KV_T, CACHE_T, KV_DTYPE>                    \
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
          head_stride, static_cast<int>(num_heads), static_cast<int>(head_size), static_cast<int>(block_size));

// Main entry point with full validation and dispatch
void commit_draft_layer(
    int64_t key_ptr,
    int64_t value_ptr,
    int64_t key_cache_ptr,
    int64_t value_cache_ptr,
    int64_t mask_ptr,
    int64_t slot_ptr,
    int64_t k_scale_ptr,
    int64_t v_scale_ptr,
    bool scale_is_per_token,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t block_stride,
    int64_t page_stride,
    int64_t head_stride,
    int64_t layout,
    const std::string& key_value_dtype,
    const std::string& kv_cache_dtype
) {
    // Issue #7: TORCH_CHECK for null pointers
    TORCH_CHECK(key_ptr != 0, "key_ptr is null");
    TORCH_CHECK(value_ptr != 0, "value_ptr is null");
    TORCH_CHECK(key_cache_ptr != 0, "key_cache_ptr is null");
    TORCH_CHECK(value_cache_ptr != 0, "value_cache_ptr is null");
    TORCH_CHECK(mask_ptr != 0, "mask_ptr is null");
    TORCH_CHECK(slot_ptr != 0, "slot_ptr is null");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_heads > 0, "num_heads must be positive");
    TORCH_CHECK(head_size > 0, "head_size must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    // Compute strides for draft tensors
    // Key/value layout: [num_tokens, num_heads, head_size]
    int64_t key_stride = num_heads * head_size;
    int64_t value_stride = num_heads * head_size;

    // Issue #4: Grid/block dimensions matching reshape_and_cache_flash
    // Cast int64_t to unsigned int for dim3 constructor
    dim3 grid(static_cast<unsigned int>(num_tokens));
    dim3 block(static_cast<unsigned int>(std::min(num_heads * head_size, static_cast<int64_t>(512))));

    // Get CUDA stream
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Issue #3: Convert key_value_dtype string to ScalarType
    at::ScalarType src_dtype;
    if (key_value_dtype == "fp16") {
        src_dtype = at::ScalarType::Half;
    } else if (key_value_dtype == "bf16") {
        src_dtype = at::ScalarType::BFloat16;
    } else if (key_value_dtype == "fp32") {
        src_dtype = at::ScalarType::Float;
    } else {
        TORCH_CHECK(false, "Unsupported key_value_dtype: ", key_value_dtype);
    }

    // Issue #3: Full dtype/cache type dispatch
    DISPATCH_BY_KV_CACHE_DTYPE(src_dtype, kv_cache_dtype, CALL_COMMIT_DRAFT_KERNEL);
}
