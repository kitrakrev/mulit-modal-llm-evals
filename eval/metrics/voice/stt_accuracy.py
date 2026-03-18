"""Speech-to-text accuracy metrics: Word Error Rate and Character Error Rate."""

from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class WordErrorRateMetric(BaseMetric):
    """Metric that computes Word Error Rate (WER) for speech-to-text evaluation.

    WER = (Substitutions + Insertions + Deletions) / Reference Word Count

    A lower score indicates better accuracy. Score is between 0.0 and 1.0+
    (can exceed 1.0 if insertions exceed reference length).
    """

    @property
    def name(self) -> str:
        return "wer"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.VOICE_STT, EvalMode.VOICE_E2E]

    @staticmethod
    def _edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int, int]:
        """Compute edit distance operations between reference and hypothesis.

        Returns:
            A tuple of (substitutions, insertions, deletions).
        """
        n = len(ref_tokens)
        m = len(hyp_tokens)

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,  # substitution
                        dp[i - 1][j] + 1,       # deletion
                        dp[i][j - 1] + 1,       # insertion
                    )

        # Backtrace to count operations
        i, j = n, m
        substitutions = insertions = deletions = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

        return substitutions, insertions, deletions

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute Word Error Rate.

        Args:
            reference: The ground-truth transcription text.
            hypothesis: The model's predicted transcription text.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to VOICE_STT).

        Returns:
            MetricResult with WER score (lower is better).
        """
        reference: str = kwargs.get("reference", kwargs.get("expected_answer", ""))
        hypothesis: str = kwargs.get("hypothesis", kwargs.get("model_answer", ""))
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.VOICE_STT)
        self.validate_mode(eval_mode)

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens:
            wer = 0.0 if not hyp_tokens else float("inf")
            return MetricResult(
                metric_name=self.name,
                score=wer,
                details={"reference_words": 0, "hypothesis_words": len(hyp_tokens)},
                model_id=model_id,
                eval_mode=eval_mode,
            )

        subs, ins, dels = self._edit_distance(ref_tokens, hyp_tokens)
        wer = (subs + ins + dels) / len(ref_tokens)

        return MetricResult(
            metric_name=self.name,
            score=wer,
            details={
                "substitutions": subs,
                "insertions": ins,
                "deletions": dels,
                "reference_words": len(ref_tokens),
                "hypothesis_words": len(hyp_tokens),
            },
            model_id=model_id,
            eval_mode=eval_mode,
        )


class CharacterErrorRateMetric(BaseMetric):
    """Metric that computes Character Error Rate (CER) for speech-to-text evaluation.

    CER = (Substitutions + Insertions + Deletions) / Reference Character Count

    A lower score indicates better accuracy.
    """

    @property
    def name(self) -> str:
        return "cer"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.VOICE_STT, EvalMode.VOICE_E2E]

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute Character Error Rate.

        Args:
            reference: The ground-truth transcription text.
            hypothesis: The model's predicted transcription text.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to VOICE_STT).

        Returns:
            MetricResult with CER score (lower is better).
        """
        reference: str = kwargs.get("reference", kwargs.get("expected_answer", ""))
        hypothesis: str = kwargs.get("hypothesis", kwargs.get("model_answer", ""))
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.VOICE_STT)
        self.validate_mode(eval_mode)

        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())

        if not ref_chars:
            cer = 0.0 if not hyp_chars else float("inf")
            return MetricResult(
                metric_name=self.name,
                score=cer,
                details={"reference_chars": 0, "hypothesis_chars": len(hyp_chars)},
                model_id=model_id,
                eval_mode=eval_mode,
            )

        # Use standard edit distance at character level
        n = len(ref_chars)
        m = len(hyp_chars)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                    )

        edit_distance = dp[n][m]
        cer = edit_distance / n

        return MetricResult(
            metric_name=self.name,
            score=cer,
            details={
                "edit_distance": edit_distance,
                "reference_chars": n,
                "hypothesis_chars": m,
            },
            model_id=model_id,
            eval_mode=eval_mode,
        )
