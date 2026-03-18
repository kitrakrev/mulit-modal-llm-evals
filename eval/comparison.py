"""Multi-model comparison engine for evaluating and ranking models."""

import uuid
from dataclasses import dataclass, field
from typing import Any

import scipy.stats as stats

from eval.metrics.base import EvalMode, MetricResult
from eval.runner import EvalRunner, EvalRunResult, EvalSample


@dataclass
class ModelRanking:
    """Ranking result for a single model."""

    model_id: str
    avg_scores: dict[str, float]
    overall_score: float
    rank: int
    metric_results: list[MetricResult] = field(default_factory=list)


@dataclass
class SignificanceTest:
    """Result of a pairwise statistical significance test."""

    model_a: str
    model_b: str
    metric_name: str
    p_value: float
    significant: bool
    effect_size: float


@dataclass
class ComparisonResult:
    """Result of a multi-model comparison."""

    comparison_id: str
    rankings: list[ModelRanking]
    significance_tests: list[SignificanceTest]
    samples_count: int
    mode: EvalMode

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the comparison.

        Returns:
            A dict with comparison metadata and ranked model results.
        """
        return {
            "comparison_id": self.comparison_id,
            "mode": self.mode.value,
            "samples_count": self.samples_count,
            "rankings": [
                {
                    "rank": r.rank,
                    "model_id": r.model_id,
                    "overall_score": r.overall_score,
                    "avg_scores": r.avg_scores,
                }
                for r in self.rankings
            ],
            "significant_differences": [
                {
                    "model_a": st.model_a,
                    "model_b": st.model_b,
                    "metric": st.metric_name,
                    "p_value": st.p_value,
                    "significant": st.significant,
                }
                for st in self.significance_tests
                if st.significant
            ],
        }


class ComparisonEngine:
    """Engine for comparing multiple models on the same dataset."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self._significance_level = significance_level

    def compare(
        self,
        model_runs: dict[str, EvalRunResult],
        comparison_id: str | None = None,
    ) -> ComparisonResult:
        """Compare multiple model evaluation runs and produce ranked results.

        Args:
            model_runs: A mapping of model_id to its EvalRunResult.
            comparison_id: Optional comparison identifier. Generated if not provided.

        Returns:
            A ComparisonResult with rankings and significance tests.
        """
        if comparison_id is None:
            comparison_id = str(uuid.uuid4())

        if not model_runs:
            raise ValueError("At least one model run is required for comparison.")

        # Aggregate metrics per model
        model_scores: dict[str, dict[str, list[float]]] = {}
        mode = EvalMode.TEXT

        for model_id, run_result in model_runs.items():
            mode = run_result.mode
            scores: dict[str, list[float]] = {}
            for m in run_result.metrics:
                scores.setdefault(m.metric_name, []).append(m.score)
            model_scores[model_id] = scores

        # Compute average scores and overall score per model
        model_avg: dict[str, dict[str, float]] = {}
        model_overall: dict[str, float] = {}

        for model_id, scores in model_scores.items():
            avg_scores = {
                name: sum(vals) / len(vals) for name, vals in scores.items()
            }
            model_avg[model_id] = avg_scores
            model_overall[model_id] = (
                sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0
            )

        # Rank models by overall score (descending)
        sorted_models = sorted(model_overall.items(), key=lambda x: x[1], reverse=True)

        rankings: list[ModelRanking] = []
        for rank, (model_id, overall) in enumerate(sorted_models, start=1):
            rankings.append(
                ModelRanking(
                    model_id=model_id,
                    avg_scores=model_avg[model_id],
                    overall_score=overall,
                    rank=rank,
                    metric_results=model_runs[model_id].metrics,
                )
            )

        # Pairwise significance tests
        significance_tests = self._compute_significance(model_scores)

        samples_count = max(
            (r.samples_count for r in model_runs.values()), default=0
        )

        return ComparisonResult(
            comparison_id=comparison_id,
            rankings=rankings,
            significance_tests=significance_tests,
            samples_count=samples_count,
            mode=mode,
        )

    def _compute_significance(
        self, model_scores: dict[str, dict[str, list[float]]]
    ) -> list[SignificanceTest]:
        """Compute pairwise statistical significance tests between models.

        Uses the Welch's t-test for comparing metric score distributions.

        Args:
            model_scores: Per-model, per-metric score lists.

        Returns:
            A list of SignificanceTest results.
        """
        results: list[SignificanceTest] = []
        model_ids = list(model_scores.keys())

        # Collect all metric names
        all_metrics: set[str] = set()
        for scores in model_scores.values():
            all_metrics.update(scores.keys())

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                model_a = model_ids[i]
                model_b = model_ids[j]

                for metric_name in all_metrics:
                    scores_a = model_scores[model_a].get(metric_name, [])
                    scores_b = model_scores[model_b].get(metric_name, [])

                    if len(scores_a) < 2 or len(scores_b) < 2:
                        continue

                    t_stat, p_value = stats.ttest_ind(
                        scores_a, scores_b, equal_var=False
                    )

                    # Cohen's d effect size
                    mean_diff = abs(
                        sum(scores_a) / len(scores_a) - sum(scores_b) / len(scores_b)
                    )
                    pooled_std = (
                        (
                            sum((x - sum(scores_a) / len(scores_a)) ** 2 for x in scores_a)
                            + sum((x - sum(scores_b) / len(scores_b)) ** 2 for x in scores_b)
                        )
                        / (len(scores_a) + len(scores_b) - 2)
                    ) ** 0.5

                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

                    results.append(
                        SignificanceTest(
                            model_a=model_a,
                            model_b=model_b,
                            metric_name=metric_name,
                            p_value=p_value,
                            significant=p_value < self._significance_level,
                            effect_size=effect_size,
                        )
                    )

        return results

    def run_comparison(
        self,
        runner: EvalRunner,
        model_ids: list[str],
        samples: list[EvalSample],
        mode: EvalMode = EvalMode.TEXT,
        comparison_id: str | None = None,
    ) -> ComparisonResult:
        """Run evaluation for multiple models and compare results.

        Args:
            runner: The EvalRunner to use for evaluation.
            model_ids: List of model identifiers to compare.
            samples: The evaluation samples.
            mode: The evaluation mode.
            comparison_id: Optional comparison identifier.

        Returns:
            A ComparisonResult with rankings and significance tests.
        """
        model_runs: dict[str, EvalRunResult] = {}
        for model_id in model_ids:
            run_result = runner.run(samples=samples, model_id=model_id, mode=mode)
            model_runs[model_id] = run_result

        return self.compare(model_runs, comparison_id=comparison_id)
