"""Tests for voice evaluation components."""

import numpy as np
import pytest

from eval.metrics.voice.stt_accuracy import CharacterErrorRateMetric, WordErrorRateMetric
from eval.metrics.voice.tts_quality import MeanOpinionScoreMetric, NaturalnessMetric
from eval.metrics.voice.voice_latency import VoiceLatencyMetric
from voice.audio_utils import AudioData


class TestWordErrorRate:
    """Tests for WordErrorRateMetric."""

    def test_perfect_transcription(self):
        metric = WordErrorRateMetric()
        result = metric.compute(
            reference="hello world this is a test",
            hypothesis="hello world this is a test",
        )
        assert result.score == 0.0

    def test_completely_wrong(self):
        metric = WordErrorRateMetric()
        result = metric.compute(
            reference="hello world",
            hypothesis="foo bar",
        )
        assert result.score == 1.0

    def test_partial_match(self):
        metric = WordErrorRateMetric()
        result = metric.compute(
            reference="the quick brown fox",
            hypothesis="the quick red fox",
        )
        assert 0.0 < result.score < 1.0
        assert result.details["substitutions"] == 1

    def test_empty_reference(self):
        metric = WordErrorRateMetric()
        result = metric.compute(reference="", hypothesis="some words")
        assert result.score == float("inf")

    def test_empty_hypothesis(self):
        metric = WordErrorRateMetric()
        result = metric.compute(reference="hello world", hypothesis="")
        assert result.score == 1.0


class TestCharacterErrorRate:
    """Tests for CharacterErrorRateMetric."""

    def test_perfect_transcription(self):
        metric = CharacterErrorRateMetric()
        result = metric.compute(reference="hello", hypothesis="hello")
        assert result.score == 0.0

    def test_single_character_error(self):
        metric = CharacterErrorRateMetric()
        result = metric.compute(reference="hello", hypothesis="hallo")
        assert result.score == pytest.approx(0.2)

    def test_empty_reference(self):
        metric = CharacterErrorRateMetric()
        result = metric.compute(reference="", hypothesis="abc")
        assert result.score == float("inf")


class TestMeanOpinionScore:
    """Tests for MeanOpinionScoreMetric."""

    def test_precomputed_score(self):
        metric = MeanOpinionScoreMetric()
        result = metric.compute(mos_score=4.2)
        assert result.score == 4.2

    def test_score_clamping(self):
        metric = MeanOpinionScoreMetric()
        result = metric.compute(mos_score=6.0)
        assert result.score == 5.0

    def test_snr_estimation(self):
        metric = MeanOpinionScoreMetric()
        result = metric.compute(signal_to_noise_ratio=25.0)
        assert 1.0 <= result.score <= 5.0

    def test_missing_input_raises(self):
        metric = MeanOpinionScoreMetric()
        with pytest.raises(ValueError):
            metric.compute()


class TestNaturalnessMetric:
    """Tests for NaturalnessMetric."""

    def test_precomputed_score(self):
        metric = NaturalnessMetric()
        result = metric.compute(naturalness_score=0.85)
        assert result.score == 0.85

    def test_prosody_estimation(self):
        metric = NaturalnessMetric()
        result = metric.compute(
            pitch_variance=50.0,
            speaking_rate=3.25,
            pause_ratio=0.225,
        )
        assert result.score == pytest.approx(1.0, abs=0.01)


class TestVoiceLatency:
    """Tests for VoiceLatencyMetric."""

    def test_precomputed_latency(self):
        metric = VoiceLatencyMetric()
        result = metric.compute(precomputed_latency_ms=150.0)
        assert result.score == 150.0

    def test_with_audio_duration(self):
        metric = VoiceLatencyMetric()
        result = metric.compute(
            precomputed_latency_ms=500.0,
            audio_duration_ms=1000.0,
        )
        assert result.details["real_time_factor"] == 0.5

    def test_inference_fn(self):
        metric = VoiceLatencyMetric()
        result = metric.compute(inference_fn=lambda: None)
        assert result.score >= 0.0

    def test_missing_input_raises(self):
        metric = VoiceLatencyMetric()
        with pytest.raises(ValueError):
            metric.compute()


class TestAudioData:
    """Tests for AudioData dataclass."""

    def test_duration_calculation(self):
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000)
        assert audio.duration_seconds == pytest.approx(1.0)

    def test_explicit_duration(self):
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration_seconds=2.0)
        assert audio.duration_seconds == 2.0
