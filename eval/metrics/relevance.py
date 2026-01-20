"""Relevance metric for measuring retrieval quality via query-context similarity."""

import math
import re
from collections import Counter
from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class RelevanceMetric(BaseMetric):
    """Metric that measures retrieval quality by computing cosine similarity
    between term frequency vectors of the query and context.
    """

    @property
    def name(self) -> str:
        return "relevance"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.TEXT, EvalMode.RAG]

    @staticmethod
    def _term_frequency(text: str) -> Counter:
        """Compute term frequency as a Counter of lowercased words."""
        return Counter(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
        """Compute cosine similarity between two term frequency vectors.

        Returns:
            Cosine similarity between 0.0 and 1.0.
        """
        if not vec_a or not vec_b:
            return 0.0

        shared_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[k] * vec_b[k] for k in shared_keys)

        magnitude_a = math.sqrt(sum(v * v for v in vec_a.values()))
        magnitude_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute relevance metric.

        Args:
            query: The user query text.
            context: The retrieved context text.
            model_id: Optional model identifier.
            mode: Optional evaluation mode (defaults to TEXT).

        Returns:
            MetricResult with cosine similarity score.
        """
        query: str = kwargs.get("query", "")
        context: str = kwargs.get("context", "")
        model_id: str = kwargs.get("model_id", "")
        mode: EvalMode = kwargs.get("mode", EvalMode.TEXT)
        self.validate_mode(mode)

        query_tf = self._term_frequency(query)
        context_tf = self._term_frequency(context)
        shared_terms = sorted(set(query_tf.keys()) & set(context_tf.keys()))

        similarity = self._cosine_similarity(query_tf, context_tf)

        return MetricResult(
            metric_name=self.name,
            score=similarity,
            details={
                "cosine_similarity": similarity,
                "query_terms": len(query_tf),
                "context_terms": len(context_tf),
                "shared_terms": shared_terms,
            },
            model_id=model_id,
            eval_mode=mode,
        )
