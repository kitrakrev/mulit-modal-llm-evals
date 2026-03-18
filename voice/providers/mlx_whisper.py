"""MLX Whisper provider for local speech-to-text."""

import time
from typing import Any

from voice.audio_utils import AudioData
from voice.stt_runner import STTResult


class MLXWhisperProvider:
    """Speech-to-text provider using MLX Whisper for local inference.

    Uses mlx-community Whisper models for on-device transcription
    on Apple Silicon hardware.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/whisper-large-v3-turbo",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._extra = kwargs
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load the MLX Whisper model, caching for reuse.

        Returns:
            The loaded Whisper model.
        """
        if self._model is None:
            import mlx_whisper

            self._model = mlx_whisper
        return self._model

    async def transcribe(self, audio: AudioData, **kwargs: Any) -> STTResult:
        """Transcribe audio using MLX Whisper.

        Args:
            audio: The audio data to transcribe.
            **kwargs: Additional parameters (language, task, etc.).

        Returns:
            An STTResult with the transcription.
        """
        whisper = self._load_model()

        language = kwargs.get("language", "en")

        start = time.perf_counter()
        result = whisper.transcribe(
            audio.samples,
            path_or_hf_repo=self._model_id,
            language=language,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return STTResult(
            text=result.get("text", "").strip(),
            latency_ms=latency_ms,
            model_id=self._model_id,
            language=language,
            metadata={"provider": "mlx_whisper"},
        )
