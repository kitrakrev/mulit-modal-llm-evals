"""Cost metric for estimating inference cost based on token usage."""

from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult

# Default pricing per 1M tokens (input, output) in USD
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (0.25, 1.25),
}


class CostMetric(BaseMetric):
    """Metric that estimates inference cost in USD based on token usage and model pricing."""

    def __init__(self, custom_pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self._pricing = dict(DEFAULT_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)

    @property
    def name(self) -> str:
        return "cost"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.TEXT, EvalMode.RAG]

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute cost metric.

        Args:
            model_id: The model identifier for pricing lookup.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            eval_mode: Optional evaluation mode (defaults to TEXT).

        Returns:
            MetricResult with estimated cost score in USD.
        """
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.TEXT)
        self.validate_mode(eval_mode)

        input_tokens: int = kwargs.get("input_tokens", 0)
        output_tokens: int = kwargs.get("output_tokens", 0)

        pricing = self._pricing.get(model_id, (0.0, 0.0))
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        total_cost = input_cost + output_cost

        return MetricResult(
            metric_name=self.name,
            score=total_cost,
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "currency": "USD",
                "model_id": model_id,
            },
            model_id=model_id,
            eval_mode=eval_mode,
        )
