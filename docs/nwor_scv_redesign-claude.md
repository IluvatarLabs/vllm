# NWOR/SCV Clean-Slate Implementation Proposal

## Executive Summary

A minimal, surgical implementation of No-Write-On-Reject (NWOR) / Staged Cache Verification (SCV) for speculative decoding that intercepts KV cache writes at a single point, stages them during verification, and commits only accepted tokens.

## Core Architecture Decision: Where to Intercept

### Option 1: At the Attention Backend ✅ (RECOMMENDED)
- Intercept at `flash_attn.py` / `xformers.py` where `reshape_and_cache_flash` is called
- This is the narrowest, most surgical point
- Single place to control all KV writes

### Option 2: At Model Layer ❌
- Modify each model's attention layer
- Too invasive, requires touching every model

### Option 3: At KV Cache Manager ❌
- Replace the KV cache manager entirely
- Too broad, breaks other subsystems

## Implementation Design

### 1. Single Point of Control - KV Cache Interceptor

```python
class KVCacheInterceptor:
    """Minimal interceptor that wraps reshape_and_cache_flash"""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        self.staging_buffer = None
        self.mode = "direct"  # or "staging"

    def write_kv(self, layer, k, v, slots):
        if self.mode == "staging" and self.staging_buffer:
            # Stage to buffer
            self.staging_buffer.add(layer, k, v, slots)
        else:
            # Direct write (normal path)
            ops.reshape_and_cache_flash(k, v,
                                       self.kv_cache[layer][0],
                                       self.kv_cache[layer][1],
                                       slots, ...)
```

### 2. Simple Staging Buffer

```python
class StagingBuffer:
    def __init__(self, max_tokens, n_layers, n_heads, head_dim):
        # Pre-allocate ONCE
        self.k_buffer = torch.zeros(n_layers, max_tokens, n_heads, head_dim)
        self.v_buffer = torch.zeros(n_layers, max_tokens, n_heads, head_dim)
        self.slots = torch.zeros(max_tokens, dtype=torch.int32)
        self.count = 0

    def add(self, layer, k, v, slot):
        # Simple indexed write
        pos = self.count
        self.k_buffer[layer, pos] = k
        self.v_buffer[layer, pos] = v
        self.slots[pos] = slot
        self.count += 1

    def commit(self, accepted_count, kv_cache):
        # Bulk write accepted portion
        for layer in range(n_layers):
            if accepted_count > 0:
                ops.reshape_and_cache_flash(
                    self.k_buffer[layer, :accepted_count],
                    self.v_buffer[layer, :accepted_count],
                    kv_cache[layer][0],
                    kv_cache[layer][1],
                    self.slots[:accepted_count], ...)
        self.reset()
```

### 3. Integration in Attention Backend

```python
# In flash_attn backend:
def forward_decode(...):
    # Detect speculation
    is_speculation = (len(query) > 1 and is_verify_phase)

    if is_speculation and nwor_enabled:
        interceptor.mode = "staging"
        interceptor.staging_buffer = get_or_create_buffer()

    # Normal attention computation
    output = flash_attn_varlen_func(...)

    # KV cache writes go through interceptor
    for layer in range(n_layers):
        interceptor.write_kv(layer, k, v, slots)

    if is_speculation and nwor_enabled:
        # Commit accepted tokens
        accepted = count_accepted_tokens(output)
        interceptor.staging_buffer.commit(accepted, kv_cache)
        interceptor.mode = "direct"
```

## Key Design Principles

1. **MINIMAL INTERCEPTION**
   - Only intercept at ONE point: the actual cache write operation
   - Don't touch model code, scheduler, or anything else
   - Single `if` statement in hot path

2. **STATELESS STAGING**
   - Buffer is just a dumb tensor container
   - No complex tracking, no per-layer lists, no None gaps
   - Reset completely between rounds

3. **FAIL-SAFE**
   - If ANYTHING goes wrong, fall back to direct writes
   - Never crash, never corrupt
   - Detection based on simple flags, not complex state machines

4. **ZERO OVERHEAD WHEN DISABLED**
   - When `mode == "direct"`, it's literally just the original function call
   - No allocations, no checks, no overhead

## Pros and Cons

### Pros
1. **Surgical**: Only touches one file, one function
2. **Simple**: No complex state machines or tracking
3. **Robust**: Can't have None gaps, can't have mismatched dimensions
4. **Fast**: Single bulk write for commits
5. **Safe**: Easy fallback to direct mode

### Cons & Mitigations

| Con | Mitigation |
|-----|------------|
| Extra memory for staging buffer | Pre-allocate once, reuse. Only ~100MB for reasonable configs |
| Needs to detect speculation | Simple flag from scheduler, already available |
| All layers share same slots | This is already true in vLLM - slots are per-token, not per-layer |
| TP coordination | Each rank has own buffer, no coordination needed |

## Potential Failure Modes & Prevention

### 1. CUDA Graph Capture
- **Issue**: Fake tensors during warmup
- **Solution**: Disable NWOR during warmup (simple flag check)

### 2. Memory Pressure
- **Issue**: Buffer allocation fails
- **Solution**: Fall back to direct mode, log warning

### 3. Acceptance Detection
- **Issue**: How to know which tokens accepted?
- **Solution**: Already computed by speculative decoding logic, just pass it down

### 4. Race Conditions
- **Issue**: Multiple workers
- **Solution**: Each worker has own buffer (thread-local)

## Why This Avoids Debug Hell

1. **Single Responsibility**: Each component does ONE thing
2. **No Hidden State**: Buffer is explicit, mode is explicit
3. **Clear Data Flow**: Stage → Commit, no complex routing
4. **Testable**: Can unit test buffer independently
5. **Observable**: Simple to add metrics/logging at intercept point

## Implementation Order

1. **Phase 1**: Interceptor with direct mode only (verify no regression)
2. **Phase 2**: Add staging buffer (verify staging works)
3. **Phase 3**: Add commit logic (verify acceptance works)
4. **Phase 4**: Add optimizations (batch commits, etc.)

## Memory Footprint

For typical configuration:
- 40 layers × 8 tokens × 32 heads × 128 dim × 2 (K+V) × 2 bytes (fp16) = ~80MB
- Negligible compared to model weights (26GB for 13B model)

## Performance Analysis

### When NWOR Helps
- Low acceptance rate (<50%)
- Long speculation windows (8+ tokens)
- Memory bandwidth bottlenecked

### When NWOR Hurts
- High acceptance rate (>90%)
- Short sequences (<20 tokens)
- Single token generation
- CPU-bound scenarios

## The Key Insight

The current implementation is too complex because it tries to:
- Route through multiple layers of abstraction
- Track complex state across components
- Handle every edge case perfectly

The clean approach:
- **Intercept at ONE point**
- **Buffer is just a tensor**
- **Commit is just a bulk write**
- **Everything else falls back to normal path**

This would be ~200 lines of code total, not thousands. And it would actually work.

## Comparison with Current Implementation

### Current (Complex)
- Multiple classes: KVWriteRouter, ShadowKV, PersistentKVWriter
- State machine: WARMUP → READY → DISABLED
- Per-layer slot mapping lists with None gaps
- Complex materialization detection
- ~1000+ lines of code

### Proposed (Simple)
- Two classes: KVCacheInterceptor, StagingBuffer
- Two modes: direct, staging
- Single tensor for all slots
- No materialization complexity
- ~200 lines of code

## Conclusion

By intercepting at the narrowest possible point and keeping the staging buffer dead simple, we can implement NWOR with minimal code, minimal overhead, and maximum reliability. The key is to resist the temptation to handle every edge case and instead fall back to the proven direct-write path when anything unexpected happens.