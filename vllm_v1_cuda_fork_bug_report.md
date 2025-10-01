# [Bug] vLLM v1 with speculative decoding crashes: CUDA initialized during module import in forked worker

## Environment

- **vLLM Version:** v0.11.1rc1.dev83+gd02c41228 (v1 engine)
- **Platform:** Linux (Ubuntu/Debian in Docker)
- **Python Version:** 3.10
- **CUDA Version:** 12.x
- **PyTorch Version:** 2.x
- **Configuration:** v1 engine with speculative_config (ngram method)

## Bug Description

When using vLLM v1 with speculative decoding enabled, the engine fails to initialize with:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

**This occurs even though vLLM has safety code (`_maybe_force_spawn()`) specifically designed to prevent this exact issue.**

## Root Cause

vLLM's safety mechanism in `vllm/utils/__init__.py:_maybe_force_spawn()` checks if CUDA is initialized before forking workers:

```python
def _maybe_force_spawn():
    # ...
    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    # ...
    if reasons:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

**However, this check fails in the following scenario:**

1. `LLM()` constructor calls `get_mp_context()` → `_maybe_force_spawn()` checks CUDA
2. At this point, CUDA is **NOT** initialized yet → check passes → uses 'fork'
3. Worker process is forked with 'fork' method
4. **Child process imports modules**, including `vllm/v1/attention/backends/flash_attn.py`
5. Line 176 in `flash_attn.py` calls `get_flash_attn_version()` **during class definition**:
   ```python
   class FlashAttentionMetadataBuilder(
       AttentionMetadataBuilder[AttentionMetadata]):
       _cg_support: AttentionCGSupport = (
           AttentionCGSupport.UNIFORM_BATCH_STATEFUL
           if get_flash_attn_version() == 3  # ← CUDA init happens HERE
           else AttentionCGSupport.UNIFORM_BATCH
       )
   ```
6. This calls → `is_fa_version_supported()` → `torch.cuda.get_device_capability()` → **CUDA init in forked child** → crash

**The problem:** CUDA initialization happens **after** the safety check but **before** the worker starts executing. It happens during module import at class definition time.

## Reproduction Steps

### Minimal Code

```python
import os
# Workaround: uncomment this line to fix
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-2-7b-hf',
    speculative_config={
        'method': 'ngram',
        'num_speculative_tokens': 5,
        'ngram_prompt_lookup_max': 5,
        'ngram_prompt_lookup_min': 2,
    },
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7
)

prompts = ['Once upon a time']
params = SamplingParams(max_tokens=20)
outputs = llm.generate(prompts, params)
```

### Conditions Required

- vLLM v1 engine (default in recent versions)
- Any `speculative_config` enabled (ngram, EAGLE, Medusa, etc.)
- Default multiprocessing method ('fork' on Linux)

### What Triggers It

The bug is triggered specifically by:
- `vllm/v1/attention/backends/flash_attn.py:176` calling `get_flash_attn_version()`
- This function checks CUDA device capability at **import time**
- With speculative decoding, this module is imported in the worker process

**Without speculative decoding:** The module may not be imported, or CUDA is already initialized in the parent before fork.

## Full Error Traceback

```
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723] EngineCore failed to start.
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723] Traceback (most recent call last):
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723]   File "/workspace/vllm/vllm/v1/engine/core.py", line 714, in run_engine_core
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723]     engine_core = EngineCoreProc(*args, **kwargs)
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723]   File "/workspace/vllm/vllm/v1/engine/core.py", line 507, in __init__
(EngineCore_DP0 pid=680) ERROR 09-30 18:32:41 [core.py:723]     super().__init__(vllm_config, executor_class, log_stats,
[... truncated middle of traceback ...]
(EngineCore_DP0 pid=680)   File "/workspace/vllm/vllm/v1/attention/backends/flash_attn.py", line 176, in FlashAttentionMetadataBuilder
(EngineCore_DP0 pid=680)     if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH
(EngineCore_DP0 pid=680)   File "/workspace/vllm/vllm/attention/utils/fa_utils.py", line 56, in get_flash_attn_version
(EngineCore_DP0 pid=680)     if not is_fa_version_supported(fa_version):
(EngineCore_DP0 pid=680)   File "/workspace/vllm/vllm/vllm_flash_attn/flash_attn_interface.py", line 55, in is_fa_version_supported
(EngineCore_DP0 pid=680)     return _is_fa2_supported(device)[0]
(EngineCore_DP0 pid=680)   File "/workspace/vllm/vllm/vllm_flash_attn/flash_attn_interface.py", line 35, in _is_fa2_supported
(EngineCore_DP0 pid=680)     if torch.cuda.get_device_capability(device)[0] < 8:
(EngineCore_DP0 pid=680)   File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 600, in get_device_capability
(EngineCore_DP0 pid=680)     prop = get_device_properties(device)
(EngineCore_DP0 pid=680)   File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 616, in get_device_properties
(EngineCore_DP0 pid=680)     _lazy_init()  # will define _get_device_properties
(EngineCore_DP0 pid=680)   File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 398, in _lazy_init
(EngineCore_DP0 pid=680)     raise RuntimeError(
(EngineCore_DP0 pid=680) RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

**Key line:** `flash_attn.py:176` during class definition calls CUDA init in forked worker.

## Workaround

Set the environment variable **before** importing vLLM:

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
# ... rest of code
```

This forces vLLM to use 'spawn' for worker processes, avoiding the fork-after-CUDA issue.

## Proposed Fixes

### Option 1: Delay CUDA Checks (Recommended)

Move the `get_flash_attn_version()` call in `flash_attn.py:176` from class definition time to runtime:

```python
class FlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AttentionMetadata]):
    _cg_support: AttentionCGSupport = None  # Computed lazily

    def __init__(self, ...):
        super().__init__(...)
        if self._cg_support is None:
            # Compute once at first instantiation
            FlashAttentionMetadataBuilder._cg_support = (
                AttentionCGSupport.UNIFORM_BATCH_STATEFUL
                if get_flash_attn_version() == 3
                else AttentionCGSupport.UNIFORM_BATCH
            )
```

This delays CUDA initialization until the worker is actually executing code, not during import.

### Option 2: Force Spawn for v1 + Speculative Decoding

Enhance `_maybe_force_spawn()` to always use spawn when speculative decoding is enabled:

```python
def _maybe_force_spawn():
    # ... existing code ...

    # Check if speculative decoding is enabled
    if vllm_config and vllm_config.speculative_config:
        reasons.append("Speculative decoding requires spawn to avoid CUDA import issues")

    if reasons:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

### Option 3: Make Safety Check More Robust

Check for **potential** CUDA initialization (torch imported) rather than only checking if CUDA is **already** initialized:

```python
def _maybe_force_spawn():
    # ... existing code ...

    if 'torch' in sys.modules and torch.cuda.is_available():
        reasons.append("PyTorch CUDA available, preventing fork to avoid init issues")

    if reasons:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

## Why This Is Critical

1. **Existing safety mechanism fails:** Users expect `_maybe_force_spawn()` to protect them, but it doesn't work for this case
2. **Silent failure:** No warning is logged; the engine just crashes during initialization
3. **Affects all speculative decoding users:** Anyone using v1 engine with spec decode will hit this
4. **Workaround is non-obvious:** Users must know about the internal `VLLM_WORKER_MULTIPROC_METHOD` env var

## Related Issues

- This is similar to but distinct from the general fork+CUDA issue
- vLLM has existing code to handle this, but it's bypassed by import-time CUDA init
- PyTorch documentation recommends always using 'spawn' with CUDA: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

## Additional Context

- The bug does NOT occur without speculative decoding (flash_attn.py may not be imported)
- Setting `VLLM_WORKER_MULTIPROC_METHOD=spawn` globally would fix it but may have other side effects
- v1 engine with speculative decoding is becoming the default, making this bug more visible

## Recommendation

**Option 1 (delay CUDA checks) is the best fix** because:
- It's a small, surgical change
- No performance impact
- Preserves existing behavior for non-spec-decode users
- Aligns with best practices (don't do expensive checks at import time)

Alternatively, **always use 'spawn' for v1 engine** would be a safe default given PyTorch's recommendations for CUDA + multiprocessing.
