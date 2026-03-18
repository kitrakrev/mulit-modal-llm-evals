"""Speech-to-text runner for evaluating STT models on audio datasets."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from voice.audio_utils import AudioData, load_audio


class STTProvider(Protocol):
    """Protocol for speech-to-text providers."""

    async def transcribe(self, audio: AudioData, **kwargs: Any) -> "STTResult": ...


@dataclass
class STTResult:
    """Result of a single STT transcription."""

    text: str
    latency_ms: float
    model_id: str
    language: str = "en"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class STTRunResult:
    """Result of an STT evaluation run."""

    run_id: str
    model_id: str
    results: list[STTResult]
    started_at: datetime
    completed_at: datetime

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the STT run."""
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


class STTRunner:
    """Runner that performs speech-to-text on audio files using configurable providers."""

    def __init__(self) -> None:
        self._providers: dict[str, STTProvider] = {}

    def register_provider(self, name: str, provider: STTProvider) -> None:
        """Register an STT provider.

        Args:
            name: The provider name.
            provider: The STT provider instance.
        """
        self._providers[name] = provider

    def get_provider(self, name: str) -> STTProvider:
        """Get a registered STT provider.

        Args:
            name: The provider name.

        Returns:
            The STT provider instance.

        Raises:
            KeyError: If the provider is not registered.
        """
        if name not in self._providers:
            raise KeyError(f"STT provider '{name}' is not registered.")
        return self._providers[name]

    async def transcribe_file(
        self,
        audio_path: str,
        provider_name: str,
        target_sample_rate: int = 16000,
        **kwargs: Any,
    ) -> STTResult:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file.
            provider_name: Name of the STT provider to use.
            target_sample_rate: Target sample rate for preprocessing.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An STTResult with the transcription.
        """
        provider = self.get_provider(provider_name)
        audio = load_audio(audio_path, target_sample_rate=target_sample_rate)
        return await provider.transcribe(audio, **kwargs)

    async def run(
        self,
        audio_paths: list[str],
        provider_name: str,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> STTRunResult:
        """Run STT on a list of audio files.

        Args:
            audio_paths: List of paths to audio files.
            provider_name: Name of the STT provider to use.
            run_id: Optional run identifier.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An STTRunResult with all transcription results.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        provider = self.get_provider(provider_name)
        started_at = datetime.now(timezone.utc)
        results: list[STTResult] = []

        for path in audio_paths:
            audio = load_audio(path)
            result = await provider.transcribe(audio, **kwargs)
            results.append(result)

        completed_at = datetime.now(timezone.utc)

        return STTRunResult(
            run_id=run_id,
            model_id=provider_name,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )
