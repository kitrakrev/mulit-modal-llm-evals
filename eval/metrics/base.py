"""Base metric classes and types for the evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvalMode(Enum):
    """Evaluation modes supported by the framework."""

    TEXT = "text"
    RAG = "rag"
    VOICE_STT = "voice_stt"
    VOICE_TTS = "voice_tts"
    VOICE_E2E = "voice_e2e"


@dataclass
class MetricResult:
    """Result of a metric computation."""

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = ""
    eval_mode: EvalMode = EvalMode.TEXT

    def to_dict(self) -> dict[str, Any]:
        """Convert the metric result to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "eval_mode": self.eval_mode.value,
        }


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this metric."""
        ...

    @property
    @abstractmethod
    def supported_modes(self) -> list[EvalMode]:
        """The evaluation modes this metric supports."""
        ...

    @abstractmethod
    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute the metric and return a result."""
        ...

    def validate_mode(self, mode: EvalMode) -> None:
        """Validate that the given mode is supported by this metric.

        Raises:
            ValueError: If the mode is not supported.
        """
        if mode not in self.supported_modes:
            raise ValueError(
                f"Mode {mode.value} is not supported by {self.name}. "
                f"Supported modes: {[m.value for m in self.supported_modes]}"
            )
