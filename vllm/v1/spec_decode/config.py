"""
Configuration for SCV and NWOR optimizations in speculative decoding
"""

from dataclasses import dataclass
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodeOptConfig:
    """Configuration for speculative decoding optimizations (SCV and NWOR)."""

    # SCV (Speculative Chunk Verify) settings
    scv_enabled: bool = False
    verify_chunk_size: int = 4

    # NWOR (No-Write-On-Reject) settings
    use_shadow_kv: bool = False

    # Debug and profiling settings
    enable_nvtx_ranges: bool = False
    debug_alloc_counters: bool = False

    @classmethod
    def from_cli_args(cls, vllm_config) -> "SpecDecodeOptConfig":
        """
        Create config from CLI arguments or environment variables.

        Args:
            vllm_config: The main vLLM config object

        Returns:
            SpecDecodeOptConfig instance
        """
        # Try to get from vllm_config first (if args were added via patch 03)
        config = cls()

        # Check for SCV settings
        if hasattr(vllm_config, 'scv_enabled'):
            config.scv_enabled = vllm_config.scv_enabled
        else:
            # Fallback to environment variable
            config.scv_enabled = os.environ.get('VLLM_SCV_ENABLED', '0') == '1'

        if hasattr(vllm_config, 'verify_chunk_size'):
            config.verify_chunk_size = vllm_config.verify_chunk_size
        else:
            config.verify_chunk_size = int(os.environ.get('VLLM_VERIFY_CHUNK_SIZE', '4'))

        # Check for NWOR settings
        if hasattr(vllm_config, 'use_shadow_kv'):
            config.use_shadow_kv = vllm_config.use_shadow_kv
        else:
            config.use_shadow_kv = os.environ.get('VLLM_USE_SHADOW_KV', '0') == '1'

        # Debug settings
        if hasattr(vllm_config, 'enable_nvtx_ranges'):
            config.enable_nvtx_ranges = vllm_config.enable_nvtx_ranges
        else:
            config.enable_nvtx_ranges = os.environ.get('VLLM_ENABLE_NVTX_RANGES', '0') == '1'

        if hasattr(vllm_config, 'debug_alloc_counters'):
            config.debug_alloc_counters = vllm_config.debug_alloc_counters
        else:
            config.debug_alloc_counters = os.environ.get('VLLM_DEBUG_ALLOC_COUNTERS', '0') == '1'

        # Log configuration
        if config.scv_enabled or config.use_shadow_kv:
            logger.info(
                "SpecDecode optimizations: SCV=%s (chunk_size=%d), NWOR=%s, NVTX=%s",
                config.scv_enabled,
                config.verify_chunk_size,
                config.use_shadow_kv,
                config.enable_nvtx_ranges
            )

        return config

    def validate(self):
        """Validate configuration settings."""
        if self.verify_chunk_size < 1:
            raise ValueError(f"verify_chunk_size must be >= 1, got {self.verify_chunk_size}")

        if self.verify_chunk_size > 16:
            logger.warning(
                "verify_chunk_size=%d is large, may reduce benefits. "
                "Recommended range: 2-8",
                self.verify_chunk_size
            )

        if self.scv_enabled and self.verify_chunk_size == 1:
            logger.warning(
                "SCV enabled with chunk_size=1, no chunking benefit. "
                "Consider increasing verify_chunk_size or disabling SCV."
            )

        return True