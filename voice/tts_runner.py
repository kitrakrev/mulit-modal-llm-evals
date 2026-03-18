"""Text-to-speech runner for evaluating TTS models."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np


class TTSProvider(Protocol):
    """Protocol for text-to-speech providers."""

    async def synthesize(self, text: str, **kwargs: Any) -> "TTSResult": ...


@dataclass
class TTSResult:
    """Result of a single TTS synthesis."""

    audio_data: np.ndarray
    sample_rate: int
    latency_ms: float
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSRunResult:
    """Result of a TTS evaluation run."""

    run_id: str
    model_id: str
    results: list[TTSResult]
    started_at: datetime
    completed_at: datetime

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the TTS run."""
        avg_latency = (
            sum(r.latency_ms for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "samples_count": len(self.results),
            "avg_latency_ms": avg_latency,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class TTSRunner:
    """Runner that performs text-to-speech using configurable providers."""

    def __init__(self) -> None:
        self._providers: dict[str, TTSProvider] = {}

    def register_provider(self, name: str, provider: TTSProvider) -> None:
        """Register a TTS provider.

        Args:
            name: The provider name.
            provider: The TTS provider instance.
        """
        self._providers[name] = provider

    def get_provider(self, name: str) -> TTSProvider:
        """Get a registered TTS provider.

        Args:
            name: The provider name.

        Returns:
            The TTS provider instance.

        Raises:
            KeyError: If the provider is not registered.
        """
        if name not in self._providers:
            raise KeyError(f"TTS provider '{name}' is not registered.")
        return self._providers[name]

    async def synthesize(
        self,
        text: str,
        provider_name: str,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize speech from text using a provider.

        Args:
            text: The input text to synthesize.
            provider_name: Name of the TTS provider to use.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A TTSResult with the synthesized audio.
        """
        provider = self.get_provider(provider_name)
        return await provider.synthesize(text, **kwargs)

    async def run(
        self,
        texts: list[str],
        provider_name: str,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> TTSRunResult:
        """Run TTS on a list of texts.

        Args:
            texts: List of texts to synthesize.
            provider_name: Name of the TTS provider to use.
            run_id: Optional run identifier.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A TTSRunResult with all synthesis results.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        provider = self.get_provider(provider_name)
        started_at = datetime.now(timezone.utc)
        results: list[TTSResult] = []

        for text in texts:
            result = await provider.synthesize(text, **kwargs)
            results.append(result)

        completed_at = datetime.now(timezone.utc)

        return TTSRunResult(
            run_id=run_id,
            model_id=provider_name,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )
