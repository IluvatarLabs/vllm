# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NWOR metrics for tracking draft commit efficiency."""

import prometheus_client

from vllm.logger import init_logger

logger = init_logger(__name__)


class NWORMetrics:
    """Prometheus metrics for NWOR draft commit tracking.

    Tracks draft token staging and acceptance to measure write savings.
    Guarded by VLLM_NWOR_EMIT_METRICS environment variable.
    """

    def __init__(
        self,
        enabled: bool,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        self.enabled = enabled
        if not enabled:
            return

        # Total draft tokens staged (deferred from immediate write)
        counter_staged = prometheus_client.Counter(
            name="vllm:nwor_tokens_staged",
            documentation="Total draft tokens staged by NWOR",
            labelnames=labelnames,
        )
        self.counter_nwor_tokens_staged = self._make_per_engine(
            counter_staged, per_engine_labelvalues
        )

        # Draft tokens committed (accepted by rejection sampling)
        counter_committed = prometheus_client.Counter(
            name="vllm:nwor_committed_tokens",
            documentation="Draft tokens accepted and committed to KV cache",
            labelnames=labelnames,
        )
        self.counter_nwor_committed_tokens = self._make_per_engine(
            counter_committed, per_engine_labelvalues
        )

        # Draft tokens rejected (not written, bandwidth saved)
        counter_rejected = prometheus_client.Counter(
            name="vllm:nwor_rejected_tokens",
            documentation="Draft tokens rejected, writes saved",
            labelnames=labelnames,
        )
        self.counter_nwor_rejected_tokens = self._make_per_engine(
            counter_rejected, per_engine_labelvalues
        )

    def _make_per_engine(
        self,
        counter: prometheus_client.Counter,
        per_engine_labelvalues: dict[int, list[str]],
    ):
        """Create a counter for each engine label value."""
        return {
            idx: counter.labels(*labelvalues)
            for idx, labelvalues in per_engine_labelvalues.items()
        }

    def observe(self, metrics: dict, engine_idx: int = 0):
        """Update counters with draft metrics from a commit.

        Args:
            metrics: Dict with num_draft_tokens, num_draft_accepted, num_draft_rejected
            engine_idx: Engine index for multi-engine setups
        """
        if not self.enabled:
            return

        self.counter_nwor_tokens_staged[engine_idx].inc(
            metrics["num_draft_tokens"]
        )
        self.counter_nwor_committed_tokens[engine_idx].inc(
            metrics["num_draft_accepted"]
        )
        self.counter_nwor_rejected_tokens[engine_idx].inc(
            metrics["num_draft_rejected"]
        )


# Global singleton
_nwor_metrics: NWORMetrics | None = None


def initialize_nwor_metrics(
    enabled: bool,
    labelnames: list[str],
    per_engine_labelvalues: dict[int, list[str]],
) -> NWORMetrics:
    """Initialize NWOR metrics singleton."""
    global _nwor_metrics
    _nwor_metrics = NWORMetrics(enabled, labelnames, per_engine_labelvalues)
    return _nwor_metrics


def get_nwor_metrics() -> NWORMetrics | None:
    """Get NWOR metrics singleton (may be None if not initialized)."""
    return _nwor_metrics
