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

    # Draft sampling settings (fix for 100% acceptance issue)
    draft_sampling_mode: str = "stochastic"  # "stochastic" or "argmax"
    draft_temperature: float = 0.9
    draft_top_p: float = 0.95  # Nucleus sampling for drafter
    draft_top_k: int = 0  # 0 = disabled

    # Draft-anchored adaptive temperature settings
    draft_q_temp_offset: float = 0.15  # Offset added to draft_temp
    draft_q_soft_temp: float = 0.50  # Soft floor to prevent ultra-cold collapse
    draft_mix_lambda_max: float = 0.05  # Tiny smoothing over baseline

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

        # Draft sampling settings
        if hasattr(vllm_config, 'draft_sampling_mode'):
            config.draft_sampling_mode = vllm_config.draft_sampling_mode
        else:
            config.draft_sampling_mode = os.environ.get('VLLM_DRAFT_SAMPLING_MODE', 'stochastic')

        if hasattr(vllm_config, 'draft_temperature'):
            config.draft_temperature = vllm_config.draft_temperature
            print(f"[TEMP_DEBUG] Got draft_temperature from vllm_config: {config.draft_temperature}", flush=True)
        else:
            config.draft_temperature = float(os.environ.get('VLLM_DRAFT_TEMPERATURE', '0.9'))
            print(f"[TEMP_DEBUG] Using default/env draft_temperature: {config.draft_temperature}", flush=True)

        if hasattr(vllm_config, 'draft_top_p'):
            config.draft_top_p = vllm_config.draft_top_p
        else:
            config.draft_top_p = float(os.environ.get('VLLM_DRAFT_TOP_P', '0.95'))

        if hasattr(vllm_config, 'draft_top_k'):
            config.draft_top_k = vllm_config.draft_top_k
        else:
            config.draft_top_k = int(os.environ.get('VLLM_DRAFT_TOP_K', '0'))

        # Draft-anchored adaptive temperature settings
        if hasattr(vllm_config, 'draft_q_temp_offset'):
            config.draft_q_temp_offset = vllm_config.draft_q_temp_offset
        else:
            config.draft_q_temp_offset = float(os.environ.get('VLLM_DRAFT_Q_TEMP_OFFSET', '0.15'))

        if hasattr(vllm_config, 'draft_q_soft_temp'):
            config.draft_q_soft_temp = vllm_config.draft_q_soft_temp
        else:
            config.draft_q_soft_temp = float(os.environ.get('VLLM_DRAFT_Q_SOFT_TEMP', '0.50'))

        if hasattr(vllm_config, 'draft_mix_lambda_max'):
            config.draft_mix_lambda_max = vllm_config.draft_mix_lambda_max
        else:
            config.draft_mix_lambda_max = float(os.environ.get('VLLM_DRAFT_MIX_LAMBDA_MAX', '0.05'))

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
                "SpecDecode optimizations: SCV=%s (chunk_size=%d), NWOR=%s, NVTX=%s, "
                "Draft sampling=%s (temp=%.2f, top_p=%.2f, top_k=%d)",
                config.scv_enabled,
                config.verify_chunk_size,
                config.use_shadow_kv,
                config.enable_nvtx_ranges,
                config.draft_sampling_mode,
                config.draft_temperature,
                config.draft_top_p,
                config.draft_top_k
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

        # Validate draft sampling settings
        if self.draft_sampling_mode not in ["stochastic", "argmax"]:
            raise ValueError(
                f"draft_sampling_mode must be 'stochastic' or 'argmax', "
                f"got '{self.draft_sampling_mode}'"
            )

        if self.draft_temperature <= 0:
            raise ValueError(f"draft_temperature must be > 0, got {self.draft_temperature}")

        if not (0.0 < self.draft_top_p <= 1.0):
            raise ValueError(f"draft_top_p must be in (0, 1], got {self.draft_top_p}")

        if self.draft_top_k < 0:
            raise ValueError(f"draft_top_k must be >= 0, got {self.draft_top_k}")

        # Warn if using argmax (will cause 100% acceptance with similar models)
        if self.draft_sampling_mode == "argmax":
            logger.warning(
                "Draft sampling mode is 'argmax' (greedy). This may cause 100%% "
                "acceptance rate with similar draft/target models, preventing "
                "NWOR/SCV optimizations from showing benefits. "
                "Consider using 'stochastic' mode instead."
            )

        return True