# NWOR & SCV Redesign Proposal

This document captures a clean-room design for No-Write-On-Reject (NWOR) and Speculative Chunk Verify (SCV), based on lessons learned from the current branch. The goal is a minimal, auditable implementation that avoids the debugging pitfalls we encountered.

---

## 1. NWOR (No-Write-On-Reject)

### Goals
- Eliminate wasted KV writes for speculative tokens that are eventually rejected.
- Preserve correctness: the persistent KV cache must always match the accepted prefix.
- Keep staging/commit overhead low relative to saved bandwidth.
- Integrate smoothly with torch.compile / CUDA graph capture.

### Architecture
- Add a `KVWriteRouter` in front of the persistent cache with states: `DISABLED`, `WARMUP`, `READY`.
- In `WARMUP`, write directly to the persistent cache (baseline behavior).
- Once all persistent KV tensors have real storage and we are outside capture, transition to `READY`.
- In `READY`, stage K/V slices in a per-layer `ShadowKV` buffer during verification windows and commit only accepted tokens.

```
 decoder attn ──► KV router ──► ┌─────────────────────────┐
                  immediate ─┬─► persistent KV cache (real)
                  defer ─────└─► ShadowKV staging buffers
```

### ShadowKV
- Maintain per-layer staging tensors `_K[layer_idx]`, `_V[layer_idx]` of shape `(max_chunk, n_heads, head_dim)`.
- Keep per-layer slot mapping lists: `_slots[layer_idx][t] = slot_tensor` (dtype `int32`).
- Provide `stage()` and `commit_to()` APIs; `materialize()` zero-fills buffers once we leave warmup.

### Staging Flow
1. Guard `k_slice`/`v_slice` using a multi-method FakeTensor detector.
2. Copy K/V slices into staging tensors; convert slot mapping to `int32` on the staging device (`copy=True`, blocking).
3. Increment a counter of staged slices; update `_len = max(_len, t+1)`.

### Commit Flow
```
accepted_len = min(accepted_len, staged_tokens)
successful_layers = 0
for each layer:
    gather K/V slices [:accepted_len]
    gather slot map [:accepted_len]
    validate slot count
    build slot tensor (stack/concat)
    convert to int32, blocking
    try append_run(); if success → successful_layers++
if successful_layers == num_layers:
    total_committed += accepted_len
    total_rejected += staged_tokens - accepted_len
else:
    total_rejected += staged_tokens  (prefix unusable)
reset _len, log layers ok / total
```

### Router Behavior
- Transition to READY only after all persistent KV tensors report real storage.
- On the first `defer()` after READY, call `shadow.materialize()` before dropping immediate mode.
- If commit throws "doesn't have storage", log, discard staged tokens, and fall back to WARMUP.

### Instrumentation
- Info logs for materialization, staging skips, per-layer commit status, and successful commits.
- Metrics: `total_staged` (actual stage calls), `total_committed`, `total_rejected`, `acceptance_rate()`.
- Debug aids: per-layer fake tensor reports (`append_run: skipping layer=…`).

---

## 2. SCV (Speculative Chunk Verify)

### Goals
- Reduce kernel launch overhead during verification by batching speculative tokens.
- Keep GPU SMs busy (higher occupancy) during token verification.
- Interoperate cleanly with NWOR (accepted prefix length drives commit).

### Flow
1. Maintain a per-request queue of speculative tokens.
2. When `chunk_size` tokens are available (or end-of-request), build a verification batch (matrix with positions, attention context, etc.).
3. Run a single fused transformer forward for the entire chunk.
4. Compute the accepted prefix length of the chunk; feed `accepted_len` to NWOR so commit happens once per chunk.
5. Repeat until queue empty.

### Safeguards
- Chunk size configurable; auto-tune based on observed acceptance.
- Fallback to chunk size 1 when sequences/acceptance are too small.
- Instrument chunk build time, chunk acceptance rate, and kernel utilization.

---

## 3. Failure Modes & Mitigations

| Failure mode                              | Mitigation                                                     |
|-------------------------------------------|----------------------------------------------------------------|
| Fake tensors staged/committed             | Comprehensive guard in `stage()`/`append_run`; fallback to WARMUP. |
| Materialize called with staged data       | Refuse to materialize when `_len > 0`.                          |
| Slot mapping mismatch/missing entries     | Per-layer validation; log and skip problematic layer.          |
| Metrics report false successes            | Update metrics only when all layers commit successfully.       |
| Chunking wastes compute (low acceptance)  | Dynamic chunk sizing; fallback to 1.                           |
| CUDA graph capture still active           | Router stays in WARMUP until `has_real_storage()` confirmed.   |

---

## 4. Projected Impact & Feasibility

| Aspect | Estimate & Rationale |
|--------|----------------------|
| **Bandwidth savings (NWOR)** | Roughly proportional to rejected fraction. With ~30% acceptance, NWOR avoids ~70% of KV writes. After staging overhead (~8%), net savings ≈ 60%. |
| **Compute savings (SCV)** | Batching verification (chunk=4-8) reduces kernel launches 4-8×. Expect verifier throughput boost of 1.3–1.7× depending on chunk utilization. |
| **NWOR overhead** | Additional copies for staging/commit add <10% work when acceptance ≤50%; payback immediate. |
| **SCV overhead** | Chunk assembly adds minor host work; fallback to scalar mode avoids latency spikes for short sequences. |
| **Probability of success** | ≈85% with the proposed guards/state machine. Remaining 15% is corner cases (unusual KV layouts or compile behavior). |
| **Risks** | Low acceptance variance, custom KV layouts, chunk size misconfiguration. Mitigated by instrumentation, dynamic knobs, and strong error logging. |

---

## 5. Summary

Building NWOR first with the router + per-layer ShadowKV gives immediate bandwidth gains and precise metrics. Once that is proven stable, layer SCV on top to reduce verification overhead. Strict tenant separation, comprehensive guards, and focused logging make the system observable and maintainable, preventing the debugging spiral we experienced.

