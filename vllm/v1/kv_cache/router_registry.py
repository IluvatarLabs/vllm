"""
Global registry for KV routers in distributed workers.
Each worker maintains its own router instance.
"""
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache.write_router import KVWriteRouter

# Global router instance per worker
_LOCAL_ROUTER: Optional["KVWriteRouter"] = None

def set_local_router(router: "KVWriteRouter") -> None:
    """Set the router for this worker process."""
    global _LOCAL_ROUTER
    _LOCAL_ROUTER = router

def get_local_router() -> Optional["KVWriteRouter"]:
    """Get the router for this worker process."""
    return _LOCAL_ROUTER

def clear_local_router() -> None:
    """Clear the router for this worker process."""
    global _LOCAL_ROUTER
    _LOCAL_ROUTER = None