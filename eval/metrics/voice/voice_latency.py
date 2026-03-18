"""Voice latency metric for measuring STT/TTS inference time."""

import time
from typing import Any, Callable

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class VoiceLatencyMetric(BaseMetric):
    """Metric that measures voice inference latency in milliseconds.

    Supports both precomputed latency values and live measurement via
    a callable inference function. Tracks time-to-first-byte (TTFB) and
    total latency separately when available.
    """

    @property
    def name(self) -> str:
        return "voice_latency"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.VOICE_STT, EvalMode.VOICE_TTS, EvalMode.VOICE_E2E]

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute voice latency metric.

        Args:
            inference_fn: A callable to time. Mutually exclusive with precomputed values.
            precomputed_latency_ms: A precomputed total latency value in milliseconds.
            ttfb_ms: Optional time-to-first-byte in milliseconds.
            audio_duration_ms: Optional audio duration for real-time factor calculation.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to VOICE_STT).

        Returns:
            MetricResult with latency score in milliseconds.

        Raises:
            ValueError: If neither inference_fn nor precomputed_latency_ms is provided.
        """
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.VOICE_STT)
        self.validate_mode(eval_mode)

        inference_fn: Callable[..., Any] | None = kwargs.get("inference_fn")
        precomputed_latency_ms: float | None = kwargs.get("precomputed_latency_ms")
        ttfb_ms: float | None = kwargs.get("ttfb_ms")
        audio_duration_ms: float | None = kwargs.get("audio_duration_ms")

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

        details: dict[str, Any] = {"unit": "ms", "total_latency_ms": latency_ms}

        if ttfb_ms is not None:
            details["ttfb_ms"] = ttfb_ms

        if audio_duration_ms is not None and audio_duration_ms > 0:
            details["real_time_factor"] = latency_ms / audio_duration_ms
            details["audio_duration_ms"] = audio_duration_ms

        return MetricResult(
            metric_name=self.name,
            score=latency_ms,
            details=details,
            model_id=model_id,
            eval_mode=eval_mode,
        )
