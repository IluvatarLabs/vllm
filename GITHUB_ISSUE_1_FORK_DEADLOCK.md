**Title:** [Critical Bug] v1 engine + speculative_config: CUDA fork deadlock

**Labels:** bug, v1, critical, speculative-decoding

---

## Bug Description

vLLM v1 engine crashes when using `speculative_config` due to CUDA being initialized in a forked subprocess. The crash is caused by `flash_attn.py` initializing CUDA during module import (at class definition time), which happens after the worker process has forked. This bypasses vLLM's existing `_maybe_force_spawn()` safety mechanism.

## Error Message

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```

## Environment

- vLLM: v0.7.x - v0.11.x (v1 engine)
- Platform: Linux (all distros, Docker and bare metal)
- Python: 3.10+
- PyTorch: 2.x with CUDA 12.x
- Default multiprocessing method: fork (Linux default)

## Reproduction

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
)

outputs = llm.generate(['Once upon a time'], SamplingParams(max_tokens=20))
```

**Expected:** LLM initializes and generates text.
**Actual:** Crash with `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

<details>
<summary>Full traceback</summary>

```
(EngineCore_DP0) ERROR: EngineCore failed to start.
Traceback (most recent call last):
  File "vllm/v1/engine/core.py", line 714, in run_engine_core
    engine_core = EngineCoreProc(*args, **kwargs)
  File "vllm/v1/engine/core.py", line 507, in __init__
    super().__init__(vllm_config, executor_class, log_stats,
  File "vllm/v1/engine/core.py", line 85, in __init__
    self.model_executor = executor_class(vllm_config)
  ...
  File "vllm/v1/attention/backends/flash_attn.py", line 176, in FlashAttentionMetadataBuilder
    if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH
  File "vllm/attention/utils/fa_utils.py", line 56, in get_flash_attn_version
    if not is_fa_version_supported(fa_version):
  File "vllm/vllm_flash_attn/flash_attn_interface.py", line 55, in is_fa_version_supported
    return _is_fa2_supported(device)[0]
  File "vllm/vllm_flash_attn/flash_attn_interface.py", line 35, in _is_fa2_supported
    if torch.cuda.get_device_capability(device)[0] < 8:
  File "torch/cuda/__init__.py", line 398, in _lazy_init
    raise RuntimeError(
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

</details>

## Root Cause

vLLM has `_maybe_force_spawn()` ([utils/__init__.py:3013](https://github.com/vllm-project/vllm/blob/main/vllm/utils/__init__.py#L3013)) that checks if CUDA is initialized and forces spawn if needed. **However, this check fails for speculative decoding:**

1. `get_mp_context()` calls `_maybe_force_spawn()`
2. CUDA not initialized yet → check passes → uses 'fork'
3. Worker process **forks**
4. Child imports `vllm.v1.attention.backends.flash_attn`
5. **Line 176** (class definition scope) calls `get_flash_attn_version()`
6. This triggers `torch.cuda.get_device_capability()` → **CUDA init in forked child**
7. ❌ CRASH

The issue is that `flash_attn.py` has CUDA initialization **at module import time** (class attribute evaluation), not at function call time:

```python
# flash_attn.py:155
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _attn_cg_support = (
        AttentionCGSupport.UNSUPPORTED
        if get_flash_attn_version() == 3  # ← CUDA init happens HERE during import
        else AttentionCGSupport.UNIFORM_BATCH
    )
```

## Workaround

Set environment variable **before** importing vLLM:

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM  # Must come AFTER env var is set
```

Or from shell:
```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python your_script.py
```

## Proposed Fixes

### Option 1: Lazy Evaluation (Recommended)

Delay CUDA check until first use:

```python
# In flash_attn.py
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _attn_cg_support = None  # Lazy init

    @classmethod
    def get_cg_support(cls):
        if cls._attn_cg_support is None:
            cls._attn_cg_support = (
                AttentionCGSupport.UNSUPPORTED
                if get_flash_attn_version() == 3
                else AttentionCGSupport.UNIFORM_BATCH
            )
        return cls._attn_cg_support
```

### Option 2: Force Spawn for Speculative Decoding

```python
# In utils/__init__.py _maybe_force_spawn()
def _maybe_force_spawn():
    reasons = []

    # Add this check
    if os.environ.get("VLLM_SPECULATIVE_CONFIG"):
        reasons.append("Speculative decoding configured")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")

    if reasons:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

### Option 3: Documentation

Document that `VLLM_WORKER_MULTIPROC_METHOD=spawn` is **required** for v1 speculative decoding.

## Impact

- **Severity:** Critical - Complete functionality blocker
- **Affected:** All v1 engine users with `speculative_config`
- **Reproduction rate:** 100%
- **Workaround:** Exists but undocumented

## Testing Notes

Tested and verified with:
- ngram prompt lookup speculative decoding ✅
- EAGLE speculative decoding ✅
- Multiple models (Llama-2, Llama-3) ✅
- Docker and bare metal ✅
- Workaround effective in all cases ✅

## References

- PyTorch multiprocessing: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
- vLLM troubleshooting (existing but insufficient): https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
