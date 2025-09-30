# NWOR Remaining Fixes - TODO

## Status: Mostly Fixed

### ✅ Fixed (Bugs #1, #3, #4, #5, #6, #9)
1. **Config introspection** (Bug #9) - Now uses vLLM helpers (TP-aware)
2. **UnboundLocalError** (Bug #1) - is_speculation defined before conditional
3. **Buffer reset logic** (Bug #6) - No longer clears token_mask between layers
4. **Layer index tracking** (Bug #3) - Auto-detects layer from token_idx pattern
5. **Per-layer cache handling** (Bug #5) - Stores and uses correct cache per layer

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

✅ ~~**Layer index tracking** (Bug #3)~~ - FIXED
✅ ~~**Store cache refs** (Bug #5)~~ - FIXED
1. **Add commit call** (Bug #2) - Core functionality (NEXT)
2. **Speculation detection** (Bug #7) - Prevents false activation
3. **Full scheduler integration** (Bug #8) - Proper lifecycle

**Estimated effort**:
- Bug #2: 2-3 hours (need to find where accepted_len is available)
- Bug #7: 30 min (change detection logic)
- Bug #8: 2-3 days (full integration with scheduler)

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