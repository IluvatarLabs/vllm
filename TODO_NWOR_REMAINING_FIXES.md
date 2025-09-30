# NWOR Remaining Fixes - TODO

## Status: Partially Fixed

### ✅ Fixed (Bugs #1, #4, #9)
1. **Config introspection** - Now uses vLLM helpers (TP-aware)
2. **UnboundLocalError** - is_speculation defined before conditional
3. **Buffer reset logic** - No longer clears token_mask between layers

### ❌ Remaining Critical Issues

---

## Bug #2: No commit() Call (CRITICAL)

**Location**: Needs to be added in `vllm/v1/worker/gpu_model_runner.py:_sample`

**Problem**: Staged tokens are never committed to KV cache. After rejection sampling determines accepted_len, we need to flush those tokens to persistent storage.

**Where to add**:
```python
# In gpu_model_runner.py, _sample() method, after line 2130:
# sampler_output.sampled_token_ids = output_token_ids

# Add here:
interceptor = get_global_interceptor()
if interceptor and interceptor.mode == "staging":
    # Get accepted length from output_token_ids
    accepted_len = len(output_token_ids)  # or calculate from spec_decode_metadata

    # TODO: Need to pass per-layer KV caches here
    # Problem: _sample doesn't have access to key_cache/value_cache
    # Options:
    #   1. Store cache references in interceptor during forward()
    #   2. Pass caches through the call stack
    #   3. Call commit from a different location where caches are available
    interceptor.commit(accepted_len, ...)  # Missing cache parameters!
```

**Architectural Issue**: The commit needs:
- KV cache references (not available in _sample)
- Per-layer caches (one for each layer)
- kv_cache_ops module (reshape_and_cache_flash)

**Possible Solutions**:
1. **Store caches in interceptor** during forward(), retrieve in commit
2. **Move commit to end of forward** pass with explicit accepted_len parameter
3. **Add callback** mechanism from sampler to attention backend

**Recommended**: Solution #1 (store cache refs during staging) is simplest for now.

---

## Bug #3: layer_idx Always 0 (CRITICAL)

**Location**: `flash_attn.py:543`

**Problem**: All layers write to buffer index 0, overwriting each other.

**Current code**:
```python
layer_idx = getattr(layer, 'layer_idx', 0)  # Always returns 0!
```

**Solutions**:
1. **During backend construction**: Iterate through `layer_names`, store mapping
   ```python
   # In FlashAttentionMetadataBuilder.__init__
   self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}
   ```

2. **Global counter in interceptor**: Track which layer is currently executing
   ```python
   # In interceptor
   self.current_layer_idx = 0

   def next_layer(self):
       idx = self.current_layer_idx
       self.current_layer_idx = (self.current_layer_idx + 1) % self.n_layers
       return idx
   ```

3. **Add layer_idx to Attention modules**: Modify vLLM's Attention class to include index

**Recommended**: Solution #2 (global counter) is least invasive.

---

## Bug #5: Commit Uses Same Cache for All Layers (CRITICAL)

**Location**: `interceptor.py:196-211`

**Problem**: commit() loops over all layers but uses the same key_cache/value_cache for each.

**Current code**:
```python
for layer_idx in range(self.n_layers):
    kv_cache_ops.reshape_and_cache_flash(
        k_accepted, v_accepted,
        key_cache, value_cache,  # SAME for all layers!
        slots, ...
    )
```

**Solution**: Pass entire kv_cache structure and index into it:
```python
def commit(self, accepted_len: int, kv_cache_ops, kv_caches_all_layers, ...):
    for layer_idx in range(self.n_layers):
        key_cache, value_cache = kv_caches_all_layers[layer_idx].unbind(0)
        kv_cache_ops.reshape_and_cache_flash(...)
```

**Challenge**: Need to capture kv_caches for all layers during forward().

---

## Bug #7: Wrong Speculation Detection (MAJOR)

**Location**: `flash_attn.py:521`

**Problem**: `num_tokens > 1` triggers on prefill too, not just speculation.

**Current code**:
```python
is_speculation = num_tokens > 1  # False positives on prefill!
```

**Solution**: Use metadata flag:
```python
# Check if attn_metadata has speculation flag
is_speculation = (hasattr(attn_metadata, 'is_spec_decode') and
                 attn_metadata.is_spec_decode)
```

**Investigation needed**: Find the correct attribute in FlashAttentionMetadata or SpecDecodeMetadata.

---

## Bug #8: Missing Scheduler Integration (ARCHITECTURAL)

**Problem**: No clear lifecycle management for enable_staging → stage → commit.

**Current flow** (BROKEN):
```
1. Layer 0 forward() → enable_staging() → stage tokens
2. Layer 1 forward() → enable_staging() [now returns True without reset] → stage tokens
3. ...
4. Layer 39 forward() → stage tokens
5. _sample() → rejection sampler determines accepted_len
6. ❌ No commit call! Staged data lost.
```

**Desired flow**:
```
1. Scheduler detects speculation start → enable_staging() ONCE
2. All layers' forward() calls → stage to buffer
3. Scheduler runs rejection sampler → gets accepted_len
4. Scheduler calls commit(accepted_len) → writes to KV cache
5. Scheduler calls disable_staging() → cleanup
```

**Implementation**:
Need hooks in scheduler/model runner:
- `before_speculation()` → enable_staging
- `after_verification(accepted_len)` → commit
- Handle per-layer cache references

---

## Implementation Priority

1. **Layer index tracking** (Bug #3) - Enables proper staging
2. **Store cache refs** - Enables commit to work
3. **Add commit call** (Bug #2) - Core functionality
4. **Fix per-layer caches** (Bug #5) - Correctness
5. **Speculation detection** (Bug #7) - Prevents false activation
6. **Full scheduler integration** (Bug #8) - Proper lifecycle

**Estimated effort**: 4-6 hours for items 1-4, 2-3 days for item 6 (full integration).

---

## Quick Fixes for Testing

To get NWOR minimally functional for testing:

```python
# In interceptor.py, add:
def store_cache_refs(self, kv_caches):
    """Store cache references for later commit."""
    self._cached_kv_caches = kv_caches

# In flash_attn.py forward(), before staging loop:
interceptor.store_cache_refs(kv_cache)

# In interceptor.py commit():
def commit(self, accepted_len, kv_cache_ops, ...):
    if not hasattr(self, '_cached_kv_caches'):
        return
    kv_cache = self._cached_kv_caches
    # ... use stored caches
```

This allows testing of the buffer logic without full architectural integration.

---

## Notes

- Buffer reset fix (completed) allows multiple layers to stage
- Config fix (completed) enables TP and various model architectures
- UnboundLocal fix (completed) prevents crashes
- Remaining issues are primarily about lifecycle coordination and cache handling
- Core StagingBuffer design is sound - issues are in integration layer