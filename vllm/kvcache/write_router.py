# vllm/kvcache/write_router.py
import torch
from vllm.attention.utils.fa_utils import reshape_and_cache_flash

class PersistentKVWriter:
    def __init__(self, kv_cache_manager, kv_cache_dtype, scales_k=None, scales_v=None):
        self.mgr = kv_cache_manager
        self.dtype = kv_cache_dtype
        self.scales_k = scales_k
        self.scales_v = scales_v

    def _tensors(self, layer:int):
        return self.mgr.key_caches[layer], self.mgr.value_caches[layer]

    @torch.no_grad()
    def append_run(self, layer:int, K:torch.Tensor, V:torch.Tensor, slot_flat:torch.Tensor):
        key_cache, value_cache = self._tensors(layer)
        reshape_and_cache_flash(K, V, key_cache, value_cache, slot_flat,
                                self.dtype, self.scales_k, self.scales_v)

class KVWriteRouter:
    """
    When deferred+armed: attention backend stages per-timestep during verify.
    After rejection sampling: commit accepted prefixes.
    """
    def __init__(self, writer, shadow):
        self.writer = writer
        self.shadow = shadow
        self.deferred = False
        self.armed = False

    def immediate(self): self.deferred = False; self.armed = False
    def defer(self):     self.deferred = True;  self.armed = False
    def is_deferred(self): return self.deferred

    @torch.no_grad()
    def begin(self, T_total:int, slot_flat:torch.Tensor, seg_lens:torch.Tensor):
        if not self.deferred: return
        self.shadow.begin(T_total, slot_flat, seg_lens)
        self.armed = True

    @torch.no_grad()
    def stage(self, layer:int, t:int, k1t:torch.Tensor, v1t:torch.Tensor):
        if self.deferred and self.armed:
            self.shadow.stage(layer, t, k1t, v1t)

    @torch.no_grad()
    def commit(self, accepted_lens: torch.Tensor):
        if self.deferred and self.armed:
            self.shadow.commit_to(self.writer, accepted_lens)
            self.armed = False