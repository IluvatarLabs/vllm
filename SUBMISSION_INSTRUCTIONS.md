# vLLM Bug Report Submission Instructions

## Files Prepared

1. **GITHUB_ISSUE_1_FORK_DEADLOCK.md** - Critical fork/spawn bug
2. **GITHUB_ISSUE_2_SHUTDOWN.md** - Process group cleanup bug

## Submission Process

### Issue #1: Critical Fork Deadlock (Submit First)

**Why first:** This is a blocker that prevents v1 speculative decoding from working at all.

**Steps:**
1. Go to: https://github.com/vllm-project/vllm/issues/new
2. Copy the **entire contents** of `GITHUB_ISSUE_1_FORK_DEADLOCK.md`
3. Paste into the issue body
4. Verify the title is: `[Critical Bug] v1 engine + speculative_config: CUDA fork deadlock`
5. Add labels (if you have permission):
   - `bug`
   - `v1`
   - `critical`
   - `speculative-decoding`
6. Click "Submit new issue"
7. **Save the issue number** (you'll need it for Issue #2)

### Issue #2: Shutdown Cleanup (Submit After #1)

**Why second:** Depends on #1 for full reproduction, but independent bug.

**Steps:**
1. Go to: https://github.com/vllm-project/vllm/issues/new
2. Copy the **entire contents** of `GITHUB_ISSUE_2_SHUTDOWN.md`
3. **BEFORE pasting:** Find the line that says `(see Issue #XXXX)`
4. Replace `#XXXX` with the issue number from Issue #1
5. Paste the updated content into the issue body
6. Verify the title is: `[Bug] v1 EngineCore missing destroy_process_group() - spurious "died unexpectedly" errors`
7. Add labels (if you have permission):
   - `bug`
   - `v1`
   - `distributed`
8. Click "Submit new issue"
9. **Save this issue number** for your NWOR PR later

### After Submission

**In Issue #2:**
- Add a comment linking to Issue #1:
  ```
  Note: This bug is independent but easier to reproduce after #XXXX (fork/spawn fix) is applied.
  ```

**In Issue #1:**
- Optionally add a comment mentioning the shutdown bug:
  ```
  Related: #YYYY - After this fork fix is applied, there's a separate shutdown bug that becomes visible.
  ```

## Expected Response

### For Issue #1 (Critical):
- Likely fast response due to severity
- May ask for additional testing
- Might propose one of the three fix options
- Could be fixed in next patch release

### For Issue #2 (Important):
- May be scheduled for next minor release
- Straightforward fix with clear patch provided
- Low risk change (adds cleanup, doesn't modify logic)

## When to Submit NWOR PR

**Wait for:**
1. ✅ Issue #1 acknowledged (response from maintainers)
2. ✅ Issue #2 acknowledged (response from maintainers)
3. ✅ NWOR testing complete (EAGLE tests, acceptance rate validation)
4. ✅ Diagnostic logs removed (clean up the temporary debug code)

**Then submit NWOR PR with:**
- Reference to both issues in PR description
- Note that NWOR requires these infrastructure fixes
- Clear testing instructions with env var workaround

## Communication Template (If Needed)

If maintainers ask for clarification:

**For Issue #1:**
> "I discovered this while implementing NWOR (No-Write-On-Reject) for speculative decoding. The issue is that `flash_attn.py` has CUDA initialization at module import time (line 176, class attribute), which runs after fork but before any of vLLM's safety checks can prevent it. The existing `_maybe_force_spawn()` function was designed to prevent this but can't detect CUDA init that happens during imports."

**For Issue #2:**
> "This bug was masked by Issue #1. After fixing the fork issue, workers start correctly but then show spurious crash messages on shutdown. The fix is straightforward - add `destroy_process_group()` to the existing `EngineCore.shutdown()` method. I've tested this extensively and confirmed it resolves the NCCL warnings and 'died unexpectedly' errors."

## Current Status

- ✅ Bug #1: Ready to submit
- ✅ Bug #2: Ready to submit
- ⏳ NWOR: Testing in progress
  - Unit tests: 7/7 passing ✅
  - Integration tests: Infrastructure issues fixed ✅
  - EAGLE tests: Pending
  - Acceptance rate validation: Pending

## Questions?

If you encounter any issues during submission or get questions from maintainers:
1. Refer to the full bug reports (vllm_bug_report_*.md) for additional details
2. Reference the GDB traces and diagnostic logs we collected
3. Point to the workaround that users can use immediately
4. Emphasize the testing we've already done

---

**Note:** Keep these files in your vllm repository for reference when submitting the NWOR PR later.
