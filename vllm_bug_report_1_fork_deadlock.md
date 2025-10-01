# [Critical Bug] vLLM v1 with speculative_config crashes: CUDA fork deadlock

## Summary

vLLM v1 engine crashes when using `speculative_config` due to CUDA being initialized in a forked subprocess. The crash occurs because `flash_attn.py` initializes CUDA during module import (at class definition time), which happens after the worker process has forked. This bypasses vLLM's existing `_maybe_force_spawn()` safety check.

**Error:**
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```

## Environment

- **vLLM Version:** v0.7.x (v1 engine)
- **Platform:** Linux (Docker, bare metal both affected)
- **Python:** 3.10+
- **PyTorch:** 2.x with CUDA 12.x
- **Multiprocessing Start Method:** fork (Linux default)

## Reproduction Steps

### Minimal Reproduction

```python
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

### Expected Behavior

LLM initializes successfully and generates text.

### Actual Behavior

Crash with full traceback:

```
(EngineCore_DP0 pid=680) ERROR: EngineCore failed to start.
(EngineCore_DP0 pid=680) Traceback (most recent call last):
  File "/workspace/vllm/vllm/v1/attention/backends/flash_attn.py", line 176
    if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH
  ...
  File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 398
    raise RuntimeError(
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```

## Root Cause Analysis

### Why vLLM's Safety Check Fails

vLLM has `_maybe_force_spawn()` (utils/__init__.py:3013) that checks if CUDA is initialized and forces spawn mode if detected:

```python
def _maybe_force_spawn():
    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

**However, this check happens too early:**

1. `get_mp_context()` is called → runs `_maybe_force_spawn()`
2. At this point, CUDA is **not** initialized yet → check passes, uses 'fork'
3. Worker process forks
4. Child process imports `vllm.v1.worker.gpu_worker`
5. This imports `vllm.v1.worker.gpu_model_runner`
6. This imports `vllm.v1.attention.backends.flash_attn`
7. **flash_attn.py line 176** (class definition scope) calls `get_flash_attn_version()`
8. This calls `torch.cuda.get_device_capability()` → **CUDA initializes in forked child**
9. CRASH

### The Core Issue

**CUDA initialization happens during module import**, not during function calls. The `flash_attn.py` module has this at class definition level:

```python
# flash_attn.py line 155
class FlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionMetadata]):

    _attn_cg_support = (
        AttentionCGSupport.UNSUPPORTED
        if get_flash_attn_version() == 3  # <-- CUDA init happens HERE
        else AttentionCGSupport.UNIFORM_BATCH
    )
```

This runs **during import**, which is **after fork** but **before any user code**.

## Workaround

Set the environment variable **before** importing vLLM:

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
# ... rest of code
```

Or from shell:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python your_script.py
```

## Proposed Fix

### Option 1: Lazy Evaluation (Recommended)

Delay the `get_flash_attn_version()` call in flash_attn.py until first use:

```python
class FlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionMetadata]):

    _attn_cg_support = None  # Lazy init

    @classmethod
    def _get_cg_support(cls):
        if cls._attn_cg_support is None:
            cls._attn_cg_support = (
                AttentionCGSupport.UNSUPPORTED
                if get_flash_attn_version() == 3
                else AttentionCGSupport.UNIFORM_BATCH
            )
        return cls._attn_cg_support
```

### Option 2: Force Spawn for Speculative Decoding

In `_maybe_force_spawn()`, add:

```python
def _maybe_force_spawn():
    reasons = []

    # Check if speculative decoding is configured
    if hasattr(self, 'vllm_config') and self.vllm_config.speculative_config:
        reasons.append("Speculative decoding requires spawn")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")

    if reasons:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

### Option 3: Documentation

Document that `VLLM_WORKER_MULTIPROC_METHOD=spawn` is **required** for:
- All v1 engine usage with speculative decoding
- Any configuration that triggers flash attention initialization

## Impact

**Critical - Blocks all v1 engine users attempting speculative decoding:**
- Affects: Any v1 workload with `speculative_config`
- Frequency: 100% reproduction rate
- Severity: Complete failure to initialize
- Workaround: Exists but not documented

## Testing

Tested with:
- ✅ ngram prompt lookup speculative decoding
- ✅ Multiple model architectures (Llama)
- ✅ Docker and bare metal environments
- ✅ Workaround verified effective

## Related Issues

This is distinct from the general "CUDA + fork" issue documented in vLLM troubleshooting. The existing safety check `_maybe_force_spawn()` was **designed** to prevent this but fails because CUDA initialization happens during import, not at a detectable time.

## References

- PyTorch multiprocessing docs: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
- vLLM troubleshooting: https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
