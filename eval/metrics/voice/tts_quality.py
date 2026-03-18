"""Text-to-speech quality metrics: Mean Opinion Score estimation and naturalness."""

from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


class MeanOpinionScoreMetric(BaseMetric):
    """Metric that estimates Mean Opinion Score (MOS) for TTS output.

    MOS is typically rated on a 1-5 scale. This metric accepts either a precomputed
    MOS value or estimates quality from signal-level features when available.
    """

    @property
    def name(self) -> str:
        return "mos"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.VOICE_TTS, EvalMode.VOICE_E2E]

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute Mean Opinion Score.

        Args:
            mos_score: A precomputed MOS score (1.0-5.0).
            signal_to_noise_ratio: Optional SNR in dB for estimation.
            sample_rate: Optional sample rate of the audio.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to VOICE_TTS).

        Returns:
            MetricResult with MOS score.
        """
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.VOICE_TTS)
        self.validate_mode(eval_mode)

        mos_score: float | None = kwargs.get("mos_score")
        snr: float | None = kwargs.get("signal_to_noise_ratio")

        if mos_score is not None:
            score = max(1.0, min(5.0, mos_score))
        elif snr is not None:
            # Simple heuristic estimation from SNR
            # Maps SNR range [0, 50] to MOS range [1.0, 5.0]
            score = max(1.0, min(5.0, 1.0 + (snr / 50.0) * 4.0))
        else:
            raise ValueError("Either 'mos_score' or 'signal_to_noise_ratio' must be provided.")

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "scale_min": 1.0,
                "scale_max": 5.0,
                "precomputed": mos_score is not None,
            },
            model_id=model_id,
            eval_mode=eval_mode,
        )


class NaturalnessMetric(BaseMetric):
    """Metric that scores the naturalness of synthesized speech.

    Evaluates how natural-sounding TTS output is on a 0.0-1.0 scale based on
    prosody features: pitch variance, speaking rate, and pause patterns.
    """

    @property
    def name(self) -> str:
        return "naturalness"

    @property
    def supported_modes(self) -> list[EvalMode]:
        return [EvalMode.VOICE_TTS, EvalMode.VOICE_E2E]

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute naturalness score.

        Args:
            naturalness_score: A precomputed naturalness score (0.0-1.0).
            pitch_variance: Optional pitch variance for estimation.
            speaking_rate: Optional speaking rate (words per second).
            pause_ratio: Optional ratio of pauses to total duration.
            model_id: Optional model identifier.
            eval_mode: Optional evaluation mode (defaults to VOICE_TTS).

        Returns:
            MetricResult with naturalness score (0.0-1.0, higher is better).
        """
        model_id: str = kwargs.get("model_id", "")
        eval_mode: EvalMode = kwargs.get("eval_mode", EvalMode.VOICE_TTS)
        self.validate_mode(eval_mode)

        naturalness_score: float | None = kwargs.get("naturalness_score")

        if naturalness_score is not None:
            score = max(0.0, min(1.0, naturalness_score))
        else:
            # Estimate from prosody features
            pitch_variance: float = kwargs.get("pitch_variance", 0.0)
            speaking_rate: float = kwargs.get("speaking_rate", 0.0)
            pause_ratio: float = kwargs.get("pause_ratio", 0.0)

            # Ideal speaking rate ~2.5-4.0 words/sec
            rate_score = 1.0 - min(1.0, abs(speaking_rate - 3.25) / 3.25)

            # Moderate pitch variance is natural (~20-80 Hz std)
            pitch_score = 1.0 - min(1.0, abs(pitch_variance - 50.0) / 50.0)

            # Natural pause ratio ~0.15-0.30
            pause_score = 1.0 - min(1.0, abs(pause_ratio - 0.225) / 0.225)

            score = max(0.0, min(1.0, (rate_score + pitch_score + pause_score) / 3.0))

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "precomputed": naturalness_score is not None,
            },
            model_id=model_id,
            eval_mode=eval_mode,
        )
