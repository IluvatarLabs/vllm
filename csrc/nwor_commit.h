/*
 * NWOR Draft Commit Kernel
 *
 * Scatters accepted draft tokens to the KV cache after speculative verification.
 * This kernel applies acceptance masks to staged draft tensors, writing only
 * accepted tokens to their final cache locations.
 */

#pragma once

#include <torch/extension.h>
#include <string>

namespace vllm {

// Log cache slots to buffer (Copy-on-Write preparation)
void log_cache_slots(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_indices,
    torch::Tensor& log_key,
    torch::Tensor& log_value,
    int64_t block_size,
    int64_t block_stride,
    int64_t page_stride,
    int64_t head_stride,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale
);

// Copy-on-Write: Restore rejected draft slots from log buffers.
// Called during commit phase to rollback rejected drafts to their pre-write state.
//
// Args:
//   log_key: Log buffer containing original key data [num_rejected, num_heads, head_size]
//   log_value: Log buffer containing original value data
//   key_cache: KV cache to restore into
//   value_cache: KV cache to restore into
//   slot_indices: Cache slot indices to restore [num_rejected]
//   block_size: KV cache block size
//   block_stride: Stride between blocks
//   page_stride: Stride between pages
//   head_stride: Stride between heads
//   kv_cache_dtype: Cache dtype
//   log_k_scale: Logged quantization scales (empty if not per-token)
//   log_v_scale: Logged quantization scales (empty if not per-token)
//
// Returns: void (operates in-place on caches)
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
);

}  // namespace vllm
