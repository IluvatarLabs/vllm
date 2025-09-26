# vllm/kvcache/route_ctx.py
from contextvars import ContextVar
from typing import Optional

_ROUTER = ContextVar("KV_ROUTER", default=None)

def set_router(router) -> object:
    return _ROUTER.set(router)

def reset_router(token: object) -> None:
    _ROUTER.reset(token)

def get_router():
    return _ROUTER.get()