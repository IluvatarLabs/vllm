# [Bug] vLLM v1 EngineCore missing destroy_process_group() causes spurious shutdown crash

## Summary

vLLM v1 engine workers report "died unexpectedly" errors after successfully completing all work. The issue occurs because `torch.distributed.destroy_process_group()` is never called before the worker exits, causing NCCL warnings and Ray to detect the process as crashed rather than cleanly exited.

**Symptoms:**
```
INFO: EngineCore.shutdown() complete
ERROR: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

**NCCL Warning:**
```
[rank0]:[W] WARNING: destroy_process_group() was not called before program exit,
which can leak resources.
```

## Environment

- **vLLM Version:** v0.7.x (v1 engine)
- **Platform:** Linux (Docker, bare metal)
- **Python:** 3.10+
- **PyTorch:** 2.x with CUDA 12.x
- **Ray:** Latest (for multiprocessing)

## Reproduction Steps

### Minimal Reproduction

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # Required, see bug #1

from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-2-7b-hf',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7
)

prompts = ['Once upon a time']
params = SamplingParams(max_tokens=20)
outputs = llm.generate(prompts, params)

print(f'✓ Generated {len(outputs)} outputs')
# Script completes successfully but logs show crash
```

### Expected Behavior

Worker exits cleanly with no errors after generation completes.

### Actual Behavior

**Output shows success:**
```
✓ Generated 1 outputs
```

**But logs show crash:**
```
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: EngineCore.shutdown() starting
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: Clearing structured output backend
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: Shutting down model executor
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: Model executor shutdown complete
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: Shutting down scheduler
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: Scheduler shutdown complete
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: EngineCore.shutdown() complete
(EngineCore_DP0 pid=747) INFO: DIAGNOSTIC: EngineCore process exiting via atexit
[rank0]:[W] WARNING: destroy_process_group() was not called before program exit
ERROR: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

## Root Cause Analysis

### The Shutdown Flow

1. User code completes → calls `llm.shutdown()` or exits
2. vLLM triggers `EngineCore.shutdown()` in worker process
3. Shutdown completes successfully:
   - ✅ Structured output backend cleared
   - ✅ Model executor shut down
   - ✅ Scheduler shut down
4. Process exits via `atexit`
5. ⚠️ PyTorch detects process group still active → NCCL warning
6. ❌ Ray sees non-clean exit → reports "died unexpectedly"

### Why It Matters

From PyTorch distributed docs:
> "Applications are expected to call `destroy_process_group()` before exiting to
> allow any destructors to be called and resources to be released."

Without this call:
- NCCL resources leak
- Process group remains in distributed registry
- Ray/multiprocessing frameworks detect abnormal exit
- Users see spurious "crash" messages for successful runs

### Missing Code

The `EngineCore.shutdown()` method (vllm/v1/engine/core.py:356) does not destroy the process group:

```python
def shutdown(self):
    logger.info("DIAGNOSTIC: EngineCore.shutdown() starting")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()

    # MISSING: torch.distributed.destroy_process_group()

    logger.info("DIAGNOSTIC: EngineCore.shutdown() complete")
```

## Proposed Fix

Add process group destruction at the end of `EngineCore.shutdown()`:

```python
# In vllm/v1/engine/core.py

# Add import at top
import torch.distributed

# In EngineCore.shutdown() method
def shutdown(self):
    logger.info("DIAGNOSTIC: EngineCore.shutdown() starting")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()

    # FIX: Explicitly destroy the default process group before exiting
    if torch.distributed.is_initialized():
        logger.info("DIAGNOSTIC: Destroying default process group")
        torch.distributed.destroy_process_group()
        logger.info("DIAGNOSTIC: Default process group destroyed")

    logger.info("DIAGNOSTIC: EngineCore.shutdown() complete")
```

### Why This Location?

- ✅ After all components shut down (executor, scheduler)
- ✅ Before process exit
- ✅ Guaranteed to run (shutdown is always called)
- ✅ Symmetric with initialization

### Alternative: Use vLLM's Helper

vLLM has `stateless_destroy_torch_distributed_process_group()` in distributed/utils.py, but it requires the process group object. Using `torch.distributed.destroy_process_group()` for the default group is simpler and more direct.

## Impact

**Medium - Affects all v1 engine users:**
- **Frequency:** 100% of v1 engine runs
- **Severity:** Cosmetic (spurious error messages) but concerning
- **Side effects:**
  - Resource leaks
  - Confusing logs/monitoring
  - Users report "crashes" for successful runs
  - May mask real crashes

## Testing

### Test Plan

1. Run any v1 workload
2. Verify completion logs show:
   ```
   INFO: DIAGNOSTIC: Destroying default process group
   INFO: DIAGNOSTIC: Default process group destroyed
   ```
3. Verify **no** NCCL warning
4. Verify **no** "died unexpectedly" error

### Tested With

- ✅ Basic generation (no speculative decoding)
- ✅ Speculative decoding (ngram prompt lookup)
- ✅ Multiple model architectures
- ✅ Different batch sizes

## Dependencies

**This fix depends on Bug Report #1** (fork/spawn issue) being addressed first:
- Without spawn fix, workers crash during startup (different bug)
- With spawn fix + this fix, workers start AND stop cleanly

## Related Issues

This is a **distributed cleanup bug**, distinct from:
- The fork/CUDA deadlock (Bug Report #1)
- The earlier vfork() hang issues (fixed by spawn)
- General multiprocessing issues

This specifically affects the **v1 engine's shutdown path** for distributed process groups.

## References

- PyTorch distributed shutdown: https://pytorch.org/docs/stable/distributed.html#shutdown
- NCCL best practices: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html
- ProcessGroupNCCL source: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1538
