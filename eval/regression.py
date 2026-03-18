"""Regression detection for monitoring metric degradation across eval runs."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from eval.runner import EvalRunResult


@dataclass
class RegressionAlert:
    """An alert for a detected metric regression."""

    alert_id: str
    model_id: str
    metric_name: str
    baseline_score: float
    current_score: float
    percent_change: float
    threshold: float
    severity: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert the alert to a dictionary."""
        return {
            "alert_id": self.alert_id,
            "model_id": self.model_id,
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "percent_change": self.percent_change,
            "threshold": self.threshold,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
        }


class RegressionDetector:
    """Detector for identifying metric regressions between eval runs.

    Compares a current evaluation run against a baseline and flags metrics
    that have degraded beyond a configurable threshold.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        """Initialize the regression detector.

        Args:
            threshold: The minimum percentage change to trigger an alert (0.05 = 5%).
            higher_is_better: Per-metric mapping indicating direction. Defaults to True
                for unknown metrics. Set to False for metrics like latency or cost.
        """
        self._threshold = threshold
        self._higher_is_better: dict[str, bool] = higher_is_better or {
            "latency": False,
            "cost": False,
        }
        self._baselines: dict[str, dict[str, float]] = {}
        self._alerts: list[RegressionAlert] = []

    def set_baseline(self, run_result: EvalRunResult) -> None:
        """Set a baseline from an evaluation run result.

        Args:
            run_result: The eval run result to use as baseline.
        """
        summary = run_result.summary()
        self._baselines[run_result.model_id] = summary["avg_scores"]

    def set_baseline_scores(self, model_id: str, scores: dict[str, float]) -> None:
        """Set baseline scores directly for a model.

        Args:
            model_id: The model identifier.
            scores: A dict mapping metric names to average scores.
        """
        self._baselines[model_id] = dict(scores)

    def detect(self, current_run: EvalRunResult) -> list[RegressionAlert]:
        """Detect regressions in the current run compared to the baseline.

        Args:
            current_run: The current evaluation run result.

        Returns:
            A list of RegressionAlert objects for metrics that degraded beyond threshold.
        """
        model_id = current_run.model_id
        baseline_scores = self._baselines.get(model_id)

        if baseline_scores is None:
            return []

        current_summary = current_run.summary()
        current_scores: dict[str, float] = current_summary["avg_scores"]
        alerts: list[RegressionAlert] = []

        for metric_name, baseline_score in baseline_scores.items():
            current_score = current_scores.get(metric_name)
            if current_score is None:
                continue

            if baseline_score == 0.0:
                percent_change = 0.0 if current_score == 0.0 else float("inf")
            else:
                percent_change = (current_score - baseline_score) / abs(baseline_score)

            higher_better = self._higher_is_better.get(metric_name, True)

            # Determine if this is a regression
            is_regression = (
                (higher_better and percent_change < -self._threshold)
                or (not higher_better and percent_change > self._threshold)
            )

            if is_regression:
                severity = self._classify_severity(abs(percent_change))
                alert = RegressionAlert(
                    alert_id=str(uuid.uuid4()),
                    model_id=model_id,
                    metric_name=metric_name,
                    baseline_score=baseline_score,
                    current_score=current_score,
                    percent_change=percent_change * 100,
                    threshold=self._threshold * 100,
                    severity=severity,
                )
                alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts

    def get_all_alerts(self) -> list[RegressionAlert]:
        """Return all detected regression alerts.

        Returns:
            A list of all RegressionAlert objects detected so far.
        """
        return list(self._alerts)

    def clear_alerts(self) -> None:
        """Clear all stored regression alerts."""
        self._alerts.clear()

    @staticmethod
    def _classify_severity(abs_percent_change: float) -> str:
        """Classify the severity of a regression based on percent change.

        Args:
            abs_percent_change: The absolute percentage change as a fraction.

        Returns:
            A severity level string: 'low', 'medium', 'high', or 'critical'.
        """
        if abs_percent_change >= 0.50:
            return "critical"
        if abs_percent_change >= 0.25:
            return "high"
        if abs_percent_change >= 0.10:
            return "medium"
        return "low"
