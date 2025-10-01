**Title:** [Bug] v1 EngineCore missing destroy_process_group() - spurious "died unexpectedly" errors

**Labels:** bug, v1, distributed

---

## Bug Description

vLLM v1 engine workers report "died unexpectedly" errors after successfully completing all work. This occurs because `torch.distributed.destroy_process_group()` is never called before the worker process exits, leading to NCCL resource leaks and Ray detecting the process as crashed rather than cleanly exited.

## Symptoms

After **successful** completion of inference:

```
✓ Speculative decoding with NWOR completed
Generated 3 outputs

INFO: EngineCore.shutdown() complete
[rank0]:[W] WARNING: destroy_process_group() was not called before program exit
ERROR: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The work completes successfully, but the shutdown is reported as a crash.

## Environment

- vLLM: v0.7.x - v0.11.x (v1 engine)
- Platform: Linux (all distros)
- Python: 3.10+
- PyTorch: 2.x with CUDA 12.x
- Ray: Latest

## Reproduction

**Note:** This bug requires the spawn multiprocessing fix (see Issue #XXXX). Without it, you'll hit a different bug during startup.

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # Required

from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-2-7b-hf',
    tensor_parallel_size=1,
)

outputs = llm.generate(['Once upon a time'], SamplingParams(max_tokens=20))
print(f'✓ Generated {len(outputs)} outputs')  # This prints successfully

# But logs show "died unexpectedly" error
```

**Expected:** Clean shutdown with no errors.

**Actual:** Work completes, but shutdown shows crash:

<details>
<summary>Full shutdown logs</summary>

```
✓ Generated 1 outputs
(EngineCore_DP0 pid=747) INFO: EngineCore.shutdown() starting
(EngineCore_DP0 pid=747) INFO: Clearing structured output backend
(EngineCore_DP0 pid=747) INFO: Shutting down model executor
(EngineCore_DP0 pid=747) INFO: Model executor shutdown complete
(EngineCore_DP0 pid=747) INFO: Shutting down scheduler
(EngineCore_DP0 pid=747) INFO: Scheduler shutdown complete
(EngineCore_DP0 pid=747) INFO: EngineCore.shutdown() complete
(EngineCore_DP0 pid=747) INFO: EngineCore process exiting via atexit
[rank0]:[W930 18:46:19] WARNING: destroy_process_group() was not called before program exit, which can leak resources.
ERROR: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

</details>

## Root Cause

The shutdown sequence is:

1. User code completes → triggers `EngineCore.shutdown()`
2. Shutdown completes successfully (executor, scheduler, etc. all clean up)
3. Process exits
4. ⚠️ PyTorch detects process group still active → **NCCL warning**
5. ❌ Ray sees non-clean exit → reports **"died unexpectedly"**

From [PyTorch distributed docs](https://pytorch.org/docs/stable/distributed.html#shutdown):

> "Applications are expected to call `destroy_process_group()` before exiting to allow any destructors to be called and resources to be released."

The `EngineCore.shutdown()` method ([vllm/v1/engine/core.py:356](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py#L356)) currently does:

```python
def shutdown(self):
    logger.info("EngineCore.shutdown() starting")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()

    # ❌ MISSING: torch.distributed.destroy_process_group()

    logger.info("EngineCore.shutdown() complete")
```

## Proposed Fix

Add process group destruction to `EngineCore.shutdown()`:

```python
# In vllm/v1/engine/core.py

# Add import at top (if not already present)
import torch.distributed

# In EngineCore.shutdown() method
def shutdown(self):
    logger.info("EngineCore.shutdown() starting")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()

    # ✅ FIX: Explicitly destroy the default process group before exiting
    if torch.distributed.is_initialized():
        logger.info("Destroying default process group")
        torch.distributed.destroy_process_group()
        logger.info("Default process group destroyed")

    logger.info("EngineCore.shutdown() complete")
```

### Why This Location?

- After all distributed components shut down (executor, scheduler)
- Before process exit
- Guaranteed to run (shutdown always called)
- Symmetric with distributed initialization

### Patch

<details>
<summary>Complete patch</summary>

```diff
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index xxx..yyy 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -18,6 +18,7 @@ from typing import Any, Callable, Optional, TypeVar, Union

 import msgspec
+import torch.distributed
 import zmq

@@ -367,6 +368,12 @@ class EngineCore:
             logger.info("DIAGNOSTIC: Shutting down scheduler")
             self.scheduler.shutdown()
             logger.info("DIAGNOSTIC: Scheduler shutdown complete")
+
+        # Explicitly destroy the default process group before exiting.
+        if torch.distributed.is_initialized():
+            logger.info("Destroying default process group")
+            torch.distributed.destroy_process_group()
+            logger.info("Default process group destroyed")

         logger.info("DIAGNOSTIC: EngineCore.shutdown() complete")
```

</details>

## Impact

- **Severity:** Medium - Affects all v1 users
- **Frequency:** 100% of v1 engine runs
- **User impact:**
  - Confusing/alarming error messages for successful runs
  - NCCL resource leaks
  - May mask real crashes
  - Monitoring/logging pollution

## Testing

With the fix applied:

1. Run any v1 workload
2. ✅ Verify logs show: `INFO: Destroying default process group`
3. ✅ Verify logs show: `INFO: Default process group destroyed`
4. ✅ Verify **no** NCCL warning
5. ✅ Verify **no** "died unexpectedly" error

Tested successfully with:
- Basic generation ✅
- Speculative decoding (ngram) ✅
- Multiple models ✅
- Various batch sizes ✅

## Dependencies

**This bug is independent but easier to reproduce after fixing Issue #XXXX** (fork/spawn bug):
- Without spawn fix: Workers crash during startup (different bug, masks this one)
- With spawn fix: Workers start successfully but show spurious shutdown errors (this bug)
- With both fixes: Workers start AND stop cleanly ✅

## Related Information

- Not related to the fork/CUDA deadlock (different bug)
- Not related to earlier vfork() hangs (different symptoms)
- Specific to v1 engine's distributed shutdown path
- Affects Ray-based and standalone deployments

## References

- PyTorch distributed shutdown: https://pytorch.org/docs/stable/distributed.html#shutdown
- NCCL best practices: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html
- ProcessGroupNCCL warning source: [PyTorch GitHub](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1538)
