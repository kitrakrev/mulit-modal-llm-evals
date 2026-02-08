"""Tests for evaluation metrics."""

import time

import pytest

from eval.metrics.base import EvalMode, MetricResult
from eval.metrics.cost import CostMetric
from eval.metrics.faithfulness import FaithfulnessMetric
from eval.metrics.latency import LatencyMetric
from eval.metrics.hallucination import HallucinationMetric
from eval.metrics.relevance import RelevanceMetric


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_to_dict(self):
        result = MetricResult(
            metric_name="test",
            score=0.95,
            details={"key": "value"},
            model_id="gpt-4o",
            eval_mode=EvalMode.TEXT,
        )
        d = result.to_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 0.95
        assert d["details"] == {"key": "value"}
        assert d["model_id"] == "gpt-4o"
        assert d["eval_mode"] == "text"
        assert "timestamp" in d

    def test_defaults(self):
        result = MetricResult(metric_name="test", score=1.0)
        assert result.model_id == ""
        assert result.eval_mode == EvalMode.TEXT
        assert result.details == {}


class TestLatencyMetric:
    """Tests for LatencyMetric."""

    def test_precomputed_latency(self):
        metric = LatencyMetric()
        result = metric.compute(precomputed_latency_ms=150.5, model_id="gpt-4o")
        assert result.metric_name == "latency"
        assert result.score == 150.5
        assert result.details["unit"] == "ms"

    def test_with_inference_fn(self):
        metric = LatencyMetric()

        def slow_fn():
            time.sleep(0.05)

        result = metric.compute(inference_fn=slow_fn)
        assert result.score >= 50.0  # at least 50ms

    def test_no_input_raises(self):
        metric = LatencyMetric()
        with pytest.raises(ValueError, match="Either 'inference_fn' or 'precomputed_latency_ms'"):
            metric.compute()

    def test_all_modes_supported(self):
        metric = LatencyMetric()
        for mode in EvalMode:
            result = metric.compute(precomputed_latency_ms=100.0, eval_mode=mode)
            assert result.eval_mode == mode


class TestCostMetric:
    """Tests for CostMetric."""

    def test_known_model(self):
        metric = CostMetric()
        result = metric.compute(
            model_id="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        assert result.metric_name == "cost"
        assert result.score > 0.0
        assert result.details["currency"] == "USD"

    def test_unknown_model_zero_cost(self):
        metric = CostMetric()
        result = metric.compute(
            model_id="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert result.score == 0.0

    def test_custom_pricing(self):
        custom = {"my-model": (1.0, 2.0)}
        metric = CostMetric(custom_pricing=custom)
        result = metric.compute(
            model_id="my-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert result.score == pytest.approx(3.0)

    def test_unsupported_mode(self):
        metric = CostMetric()
        with pytest.raises(ValueError, match="not supported"):
            metric.compute(
                model_id="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                eval_mode=EvalMode.VOICE_STT,
            )


class TestFaithfulnessMetric:
    """Tests for FaithfulnessMetric."""

    def test_perfect_overlap(self):
        metric = FaithfulnessMetric()
        result = metric.compute(
            answer="the cat sat on the mat",
            context="the cat sat on the mat",
        )
        assert result.score == pytest.approx(1.0)
        assert result.details["unsupported_claims_count"] == 0

    def test_no_overlap(self):
        metric = FaithfulnessMetric()
        result = metric.compute(
            answer="hello world",
            context="foo bar baz",
        )
        assert result.score == pytest.approx(0.0)

    def test_partial_overlap(self):
        metric = FaithfulnessMetric()
        result = metric.compute(
            answer="the cat is happy",
            context="the cat sat on the mat",
        )
        assert 0.0 < result.score < 1.0
        assert result.details["overlap_count"] > 0
        assert result.details["unsupported_claims_count"] > 0

    def test_empty_answer(self):
        metric = FaithfulnessMetric()
        result = metric.compute(
            answer="",
            context="the cat sat on the mat",
        )
        assert result.score == pytest.approx(0.0)

    def test_empty_context(self):
        metric = FaithfulnessMetric()
        result = metric.compute(
            answer="the cat sat on the mat",
            context="",
        )
        assert result.score == pytest.approx(0.0)


class TestRelevanceMetric:
    """Tests for RelevanceMetric."""

    def test_identical_texts(self):
        metric = RelevanceMetric()
        result = metric.compute(
            query="machine learning algorithms",
            context="machine learning algorithms",
        )
        assert result.score == pytest.approx(1.0)

    def test_no_overlap(self):
        metric = RelevanceMetric()
        result = metric.compute(
            query="hello world",
            context="foo bar baz",
        )
        assert result.score == pytest.approx(0.0)

    def test_partial_relevance(self):
        metric = RelevanceMetric()
        result = metric.compute(
            query="machine learning",
            context="machine learning is a subset of artificial intelligence",
        )
        assert 0.0 < result.score < 1.0
        assert len(result.details["shared_terms"]) > 0

    def test_empty_query(self):
        metric = RelevanceMetric()
        result = metric.compute(
            query="",
            context="some context text",
        )
        assert result.score == pytest.approx(0.0)


class TestHallucinationMetric:
    """Tests for HallucinationMetric."""

    def test_fully_supported(self):
        metric = HallucinationMetric()
        result = metric.compute(
            answer="The cat sat on the mat. The mat was red.",
            context="The cat sat on the mat. The mat was red and soft.",
        )
        assert result.score == pytest.approx(1.0)
        assert result.details["hallucinated_count"] == 0

    def test_fully_hallucinated(self):
        metric = HallucinationMetric()
        result = metric.compute(
            answer="Elephants can fly through space. They breathe fire.",
            context="The cat sat on the mat.",
        )
        assert result.score < 1.0
        assert result.details["hallucinated_count"] > 0

    def test_partial_hallucination(self):
        metric = HallucinationMetric()
        result = metric.compute(
            answer="The cat sat on the mat. Elephants can fly through space.",
            context="The cat sat on the mat and was happy.",
        )
        assert 0.0 < result.score < 1.0
        assert result.details["supported_count"] > 0
        assert result.details["hallucinated_count"] > 0

    def test_empty_answer(self):
        metric = HallucinationMetric()
        result = metric.compute(
            answer="",
            context="The cat sat on the mat.",
        )
        assert result.score == pytest.approx(1.0)
        assert result.details["total_sentences"] == 0

    def test_custom_threshold(self):
        metric = HallucinationMetric(support_threshold=0.5)
        assert metric.support_threshold == 0.5
