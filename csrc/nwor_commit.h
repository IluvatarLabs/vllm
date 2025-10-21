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

// Commit draft KV tensors for one layer based on acceptance mask.
// This is called once per layer during speculative decoding commit phase.
//
// Args:
//   key_ptr: Pointer to draft key tensor [num_tokens, num_heads, head_size]
//   value_ptr: Pointer to draft value tensor
//   key_cache_ptr: Pointer to key cache [layout-dependent, see cache.h]
//   value_cache_ptr: Pointer to value cache
//   mask_ptr: Pointer to bool mask [num_tokens] indicating accepted tokens
//   slot_ptr: Pointer to int32 slot_mapping [num_tokens]
//   k_scale_ptr: Pointer to quantization scale (0 if None)
//   v_scale_ptr: Pointer to quantization scale (0 if None)
//   scale_is_per_token: Whether scales are per-token or scalar
//   num_tokens: Number of draft tokens
//   num_heads: Number of attention heads
//   head_size: Head dimension
//   block_size: KV cache block size
//   block_stride: Stride between blocks in cache
//   page_stride: Stride between pages in cache
//   head_stride: Stride between heads in cache
//   layout: CacheLayout enum (0=Flash, 1=Paged)
//   key_value_dtype: Source tensor dtype ("fp16", "bf16", "fp32")
//   kv_cache_dtype: Cache dtype ("auto", "fp8", or "fp8_e5m2")
//
// Returns: void (operates in-place on caches)
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
);
