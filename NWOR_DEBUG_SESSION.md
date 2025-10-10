# NWOR Debugging Session - Context for Next Iteration

## Mission
We're implementing **NWOR (No-Write-On-Reject)** device-side staging for EAGLE speculative decoding in vLLM. The goal: eliminate wasted memory bandwidth by staging draft token KV writes on GPU and only committing accepted tokens.

## Current Status

### ✅ What's Working
1. **Operators compile and register** in `torch.ops._C_cache_ops`:
   - `stage_kv_cache()` - Stages draft KV writes to GPU buffers
   - `commit_staged_kv_cache()` - Commits accepted tokens to main cache

2. **Atomic increment bug FIXED** (commit 6cb2e2572):
   - Was: All 512 threads executed `atomicAdd()` → 4096 staged instead of 8
   - Now: Only thread 0 atomicAdds, broadcasts via `__shared__ int`

3. **Layer registration refactored**:
   - New helper: `_nwor_utils.py::reset_nwor_staging_buffers()`
   - Uses `_get_nwor_staging_caches()` to walk actual layer structure
   - Allocates staging buffers matching current layer layout

### ❌ What's Broken
**Error when running with TP=2 + EAGLE:**
```
ERROR: NWOR fatal error: staged tokens 0 < accepted 6 for layer 32
```

**Analysis:**
- With TP=2, each rank should handle 16 layers (32 total / 2)
- EAGLE draft model adds extra layers beyond base 32
- Staging buffers allocated for ~16 layers
- Commit loop tries to access layer 32 → out of bounds

**Root cause hypothesis:**
Draft model layers are added AFTER staging buffer allocation, causing mismatch between:
- `len(staging_buffers.metadata)` = 16
- `len(layer_key_caches)` = 33+

## The Test

### Command
```bash
cd ~/Documents/GitHub/fastserve

docker run --rm --gpus all \
  -v ~/Documents/GitHub/vllm:/workspace/vllm \
  -v "$(pwd)":/workspace/fastserve \
  -w /workspace/fastserve \
  -e VLLM_USE_PRECOMPILED=0 \
  -e VLLM_TARGET_DEVICE=cuda \
  --entrypoint=/bin/bash \
  vllm-nwor:latest -lc './test_nwor_complete.sh 2>&1 | tee nwor_test_fixed.log'
```

### What to Look For
✅ **Success signals:**
- Baseline and NWOR modes both complete
- No "layer 32" errors
- Staging metadata shows reasonable token counts (not 4096)
- Acceptance rate > 0%

❌ **Failure signals:**
- `NWOR fatal error: staged tokens 0 < accepted X for layer Y`
- `RuntimeError: layer index out of bounds`
- Crash during commit phase

## Files Changed Since Last Session

### Committed (6cb2e2572)
- `csrc/cache_kernels.cu` - Atomic increment fix
- `csrc/torch_bindings.cpp` - Namespace fix
- `cmake/utils.cmake` - Build config

### Uncommitted (layer mismatch fix)
- `vllm/v1/worker/gpu_model_runner.py` - Uses new helper
- `vllm/v1/worker/_nwor_utils.py` - Staging buffer allocation helper
- `vllm/platforms/__init__.py` - Version detection fix
- `docker/Dockerfile` - Runtime dependencies
- `NWOR_IMPLEMENTATION_SUMMARY.md` - Updated docs

## Your Role in Next Session

### 1. Analyze the Test Log
When I share `nwor_test_fixed.log`, look for:
- Exact error messages and line numbers
- Staging buffer allocation logs (capacity, layer count)
- Layer registration logs (how many layers registered)
- The values in staging metadata at failure time

### 2. Diagnose Root Cause
Present findings in this format:
```
**Error:** [exact error text]

**Evidence:**
- Log line X shows: [fact]
- Log line Y shows: [fact]
- Code at file:line does: [behavior]

**Root cause:** [logical conclusion from evidence]

**Why the current fix doesn't work:** [analysis]
```

### 3. Propose Surgical Fix
**Format:**
```
**Fix approach:** [1-2 sentence summary]

**Changes required:**
1. File: path/to/file.py
   - Line X: Change [old] to [new]
   - Rationale: [why]

2. File: path/to/other.py
   - Line Y: Add [what]
   - Rationale: [why]

**Risks:** [what could break]
**Validation:** [how to verify it works]
```

### 4. WAIT FOR APPROVAL
**DO NOT:**
- Make edits without discussing first
- Implement fixes immediately
- Use "I'll just..." or "let me quickly..."

**DO:**
- Surface complete diagnosis
- Explain reasoning step-by-step
- Wait for "go ahead" before coding

## Key Architecture Notes

### Staging Flow
1. `enable_staging()` called when spec_decode_metadata present
2. Each layer: `stage_layer_writes()` → `torch.ops._C_cache_ops.stage_kv_cache()`
3. Metadata tracks: `[staged_count, error_flag]` per layer
4. After sampling: `commit_window(accepted_len)` → `torch.ops._C_cache_ops.commit_staged_kv_cache()`

### Layer Registration
- `initialize_kv_cache()` allocates staging buffers
- Calls `_get_nwor_staging_caches()` to collect ALL layers including draft
- Builds `layer_key_caches` and `cache_ptr_to_layer` mapping
- Staging buffer count MUST match layer count

### Critical Invariant
```python
len(staging_buffers.metadata) == len(layer_key_caches)
```
If violated → "layer X" out of bounds error

## Environment Details
- **Model:** Llama-2-7b-chat (32 layers total)
- **Draft:** EAGLE-llama2-chat-7B (adds 4+ layers)
- **TP:** 2 ranks (16 layers each expected)
- **Spec tokens:** 8 per request
- **Batch size:** 16 requests
- **Expected staging:** ~144 tokens/step (16 requests × 9 tokens)

## Success Criteria
1. Test completes without crashes
2. Baseline and NWOR modes both run
3. Staging metadata shows correct counts
4. Acceptance rate measured accurately
5. No layer index errors

---

**Next action:** Run the test, share the log, wait for diagnosis before any code changes.