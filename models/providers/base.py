"""Base provider interface for model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Response from a model provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base class for all model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The name of this provider."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ModelResponse with the generated text and metadata.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is healthy and reachable.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        return True
