"""Hallucination detection metric for measuring unsupported claims."""

import re
from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class HallucinationMetric(BaseMetric):
    """Metric that detects claims in an answer not supported by the provided context.

    Uses sentence-level analysis to identify hallucinated content by measuring
    token overlap between each sentence and the context. Sentences with overlap
    below the support threshold are considered hallucinated.

    Score: 1.0 = no hallucination, 0.0 = fully hallucinated.
    """

    def __init__(self, support_threshold: float = 0.3) -> None:
        """Initialize the hallucination metric.

        Args:
            support_threshold: Minimum fraction of sentence tokens that must
                appear in the context for a sentence to be considered supported.
        """
        self.support_threshold = support_threshold

    @property
    def name(self) -> str:
        return "hallucination"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.TEXT, EvalMode.RAG]

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Split text into sentences using regex on .!? delimiters.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of non-empty sentence strings.
        """
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _token_overlap(sentence: str, context: str) -> float:
        """Compute the fraction of sentence tokens present in the context.

        Args:
            sentence: A single sentence from the answer.
            context: The reference context.

        Returns:
            Fraction of sentence tokens found in context (0.0 to 1.0).
        """
        sentence_tokens = set(re.findall(r"\w+", sentence.lower()))
        if not sentence_tokens:
            return 1.0
        context_tokens = set(re.findall(r"\w+", context.lower()))
        overlap = sentence_tokens & context_tokens
        return len(overlap) / len(sentence_tokens)

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute hallucination detection metric.

        Args:
            answer: The generated answer text.
            context: The reference context the answer should be grounded in.
            model_id: Optional model identifier.
            mode: Optional evaluation mode (defaults to TEXT).

        Returns:
            MetricResult with hallucination score (1.0 = no hallucination).
        """
        answer: str = kwargs.get("answer", "")
        context: str = kwargs.get("context", "")
        model_id: str = kwargs.get("model_id", "")
        mode: EvalMode = kwargs.get("mode", EvalMode.TEXT)
        self.validate_mode(mode)

        sentences = self._sentence_split(answer)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={
                    "total_sentences": 0,
                    "supported_count": 0,
                    "hallucinated_count": 0,
                    "hallucinated_sentences": [],
                },
                model_id=model_id,
                eval_mode=mode,
            )

        hallucinated_sentences: list[str] = []
        supported_count = 0

        for sentence in sentences:
            overlap = self._token_overlap(sentence, context)
            if overlap >= self.support_threshold:
                supported_count += 1
            else:
                hallucinated_sentences.append(sentence)

        hallucinated_count = total_sentences - supported_count
        score = supported_count / total_sentences

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "total_sentences": total_sentences,
                "supported_count": supported_count,
                "hallucinated_count": hallucinated_count,
                "hallucinated_sentences": hallucinated_sentences[:10],
            },
            model_id=model_id,
            eval_mode=mode,
        )
