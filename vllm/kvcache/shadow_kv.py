# vllm/kvcache/shadow_kv.py
import torch
from typing import List, Optional

class ShadowKV:
    """
    Stage per-layer K/V for the verify window across the whole batch.
    Window is flattened: T_total = sum_i T_i. We keep slot_mapping(flat) and
    per-request segment lengths so we can commit accepted prefixes only.
    """
    def __init__(self, n_layers:int, n_kv_heads:int, head_dim:int,
                 max_tokens:int, device="cuda", dtype=torch.float16):
        self.L, self.H, self.D = n_layers, n_kv_heads, head_dim
        self.Tmax, self.dev, self.dtype = max_tokens, device, dtype
        self.K: List[torch.Tensor] = [
            torch.empty((max_tokens, n_kv_heads, head_dim), device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.V: List[torch.Tensor] = [
            torch.empty((max_tokens, n_kv_heads, head_dim), device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.T_total: int = 0
        self.slot_flat: Optional[torch.Tensor] = None
        self.stride: int = 0
        self.seg_lens_cpu: Optional[torch.Tensor] = None   # [B] int32
        self.seg_off_cpu: Optional[torch.Tensor] = None    # [B+1] int32

    @torch.no_grad()
    def begin(self, T_total:int, slot_flat:torch.Tensor, seg_lens:torch.Tensor):
        self.T_total = T_total
        self.slot_flat = slot_flat.contiguous()
        total = self.slot_flat.numel()
        if T_total == 0 or total % T_total != 0:
            raise RuntimeError(f"ShadowKV: bad slot length={total}, T_total={T_total}")
        self.stride = total // T_total
        self.seg_lens_cpu = seg_lens.to(device="cpu", dtype=torch.int32)
        self.seg_off_cpu = torch.empty(self.seg_lens_cpu.numel() + 1, dtype=torch.int32)
        torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int32), self.seg_lens_cpu]),
                     dim=0, out=self.seg_off_cpu)

    @torch.no_grad()
    def stage(self, layer:int, t:int, k1t:torch.Tensor, v1t:torch.Tensor):
        self.K[layer][t:t+1].copy_(k1t)
        self.V[layer][t:t+1].copy_(v1t)

    @torch.no_grad()
    def _slot_for_indices(self, idx: torch.Tensor) -> torch.Tensor:
        pieces = []
        for t in idx.tolist():
            s = int(t) * self.stride
            pieces.append(self.slot_flat[s:s+self.stride])
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def commit_to(self, writer, accepted_lens: torch.Tensor):
        if accepted_lens.numel() == 0: return
        acc = accepted_lens.to(device="cpu", dtype=torch.int32)
        if acc.numel() + 1 != self.seg_off_cpu.numel():
            raise RuntimeError("ShadowKV: accepted_lens size mismatch")
        idx_list = []
        for i in range(acc.numel()):
            off = int(self.seg_off_cpu[i].item())
            m   = int(acc[i].item())
            if m > 0:
                idx_list.append(torch.arange(off, off + m, dtype=torch.int32))
        if not idx_list: return
        idx = torch.cat(idx_list, dim=0).to(device=self.K[0].device)
        slot_sel = self._slot_for_indices(idx).to(device=self.K[0].device)
        for l in range(self.L):
            K_sel = self.K[l].index_select(0, idx)
            V_sel = self.V[l].index_select(0, idx)
            writer.append_run(l, K_sel, V_sel, slot_sel)