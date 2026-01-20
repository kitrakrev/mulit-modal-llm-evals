"""Faithfulness metric for measuring answer grounding in context."""

import re
from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class FaithfulnessMetric(BaseMetric):
    """Metric that measures how well an answer is grounded in the provided context.

    Uses token-level F1 score between the answer and context to quantify
    how faithfully the answer reflects the source material.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.TEXT, EvalMode.RAG]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of lowercased words."""
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _compute_f1(prediction_tokens: set[str], reference_tokens: set[str]) -> float:
        """Compute F1 score between prediction and reference token sets.

        Returns:
            F1 score between 0.0 and 1.0.
        """
        if not prediction_tokens or not reference_tokens:
            return 0.0

        overlap = prediction_tokens & reference_tokens
        if not overlap:
            return 0.0

        precision = len(overlap) / len(prediction_tokens)
        recall = len(overlap) / len(reference_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute faithfulness metric.

        Args:
            answer: The generated answer text.
            context: The reference context the answer should be grounded in.
            model_id: Optional model identifier.
            mode: Optional evaluation mode (defaults to TEXT).

        Returns:
            MetricResult with faithfulness F1 score.
        """
        answer: str = kwargs.get("answer", "")
        context: str = kwargs.get("context", "")
        model_id: str = kwargs.get("model_id", "")
        mode: EvalMode = kwargs.get("mode", EvalMode.TEXT)
        self.validate_mode(mode)

        answer_tokens = self._tokenize(answer)
        context_tokens = self._tokenize(context)
        overlap = answer_tokens & context_tokens
        unsupported_tokens = answer_tokens - context_tokens

        f1_score = self._compute_f1(answer_tokens, context_tokens)

        return MetricResult(
            metric_name=self.name,
            score=f1_score,
            details={
                "f1_score": f1_score,
                "answer_token_count": len(answer_tokens),
                "context_token_count": len(context_tokens),
                "overlap_count": len(overlap),
                "unsupported_claims_count": len(unsupported_tokens),
                "unsupported_tokens": sorted(unsupported_tokens)[:20],
            },
            model_id=model_id,
            eval_mode=mode,
        )
