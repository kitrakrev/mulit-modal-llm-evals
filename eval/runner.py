"""Evaluation runner for executing metrics against model outputs."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from eval.metrics.base import BaseMetric, EvalMode, MetricResult


@dataclass
class EvalSample:
    """A single evaluation sample."""

    query: str
    expected_answer: str
    context: str = ""
    model_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRunResult:
    """Result of an evaluation run."""

    run_id: str
    model_id: str
    mode: EvalMode
    metrics: list[MetricResult]
    samples_count: int
    started_at: datetime
    completed_at: datetime

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the evaluation run.

        Returns:
            A dict with run metadata and averaged metric scores.
        """
        metric_scores: dict[str, list[float]] = {}
        for m in self.metrics:
            metric_scores.setdefault(m.metric_name, []).append(m.score)

        avg_scores = {
            name: sum(scores) / len(scores) for name, scores in metric_scores.items()
        }

        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "mode": self.mode.value,
            "samples_count": self.samples_count,
            "avg_scores": avg_scores,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class EvalRunner:
    """Runner that evaluates model outputs against configured metrics."""

    def __init__(self) -> None:
        self._metrics: list[BaseMetric] = []

    def add_metric(self, metric: BaseMetric) -> None:
        """Add a metric to the runner.

        Args:
            metric: The metric to add.
        """
        self._metrics.append(metric)

    def evaluate_sample(
        self, sample: EvalSample, mode: EvalMode = EvalMode.TEXT
    ) -> list[MetricResult]:
        """Evaluate a single sample against all configured metrics.

        Args:
            sample: The evaluation sample.
            mode: The evaluation mode.

        Returns:
            A list of MetricResult objects.
        """
        results: list[MetricResult] = []
        for metric in self._metrics:
            if mode in metric.supported_modes:
                result = metric.compute(
                    query=sample.query,
                    expected_answer=sample.expected_answer,
                    model_answer=sample.model_answer,
                    context=sample.context,
                )
                results.append(result)
        return results

    def run(
        self,
        samples: list[EvalSample],
        model_id: str = "unknown",
        mode: EvalMode = EvalMode.TEXT,
        run_id: str | None = None,
    ) -> EvalRunResult:
        """Run evaluation on a list of samples.

        Args:
            samples: The evaluation samples.
            model_id: Identifier for the model being evaluated.
            mode: The evaluation mode.
            run_id: Optional run identifier. Generated if not provided.

        Returns:
            An EvalRunResult with all metric results.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        started_at = datetime.now(timezone.utc)
        all_metrics: list[MetricResult] = []

        for sample in samples:
            sample_results = self.evaluate_sample(sample, mode=mode)
            all_metrics.extend(sample_results)

        completed_at = datetime.now(timezone.utc)

        return EvalRunResult(
            run_id=run_id,
            model_id=model_id,
            mode=mode,
            metrics=all_metrics,
            samples_count=len(samples),
            started_at=started_at,
            completed_at=completed_at,
        )
