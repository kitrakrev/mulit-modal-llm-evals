"""Latency metric for measuring inference time."""

import time
from typing import Any, Callable

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class LatencyMetric(BaseMetric):
    """Metric that measures inference latency in milliseconds."""

    @property
    def name(self) -> str:
        return "latency"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return list(EvalMode)

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute latency metric.

        Args:
            inference_fn: A callable to time. Mutually exclusive with precomputed_latency_ms.
            precomputed_latency_ms: A precomputed latency value in milliseconds.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to TEXT).

        Returns:
            MetricResult with latency score in milliseconds.

        Raises:
            ValueError: If neither inference_fn nor precomputed_latency_ms is provided.
        """
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.TEXT)
        self.validate_mode(eval_mode)

        inference_fn: Callable[..., Any] | None = kwargs.get("inference_fn")
        precomputed_latency_ms: float | None = kwargs.get("precomputed_latency_ms")

        if precomputed_latency_ms is not None:
            latency_ms = precomputed_latency_ms
        elif inference_fn is not None:
            start = time.perf_counter()
            inference_fn()
            end = time.perf_counter()
            latency_ms = (end - start) * 1000.0
        else:
            raise ValueError(
                "Either 'inference_fn' or 'precomputed_latency_ms' must be provided."
            )

        return MetricResult(
            metric_name=self.name,
            score=latency_ms,
            details={"unit": "ms"},
            model_id=model_id,
            eval_mode=eval_mode,
        )
