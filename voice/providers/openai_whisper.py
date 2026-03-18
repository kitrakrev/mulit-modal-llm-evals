"""OpenAI Whisper provider for cloud speech-to-text."""

import io
import os
import time
from typing import Any

from voice.audio_utils import AudioData
from voice.stt_runner import STTResult


class OpenAIWhisperProvider:
    """Speech-to-text provider using the OpenAI Whisper API.

    Uses the OpenAI audio transcription endpoint for cloud-based STT.
    """

    def __init__(
        self,
        model_id: str = "whisper-1",
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._extra = kwargs
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI async client.

        Returns:
            An AsyncOpenAI client instance.

        Raises:
            RuntimeError: If the API key environment variable is not set.
        """
        if self._client is None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable '{self._api_key_env}' is not set."
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def transcribe(self, audio: AudioData, **kwargs: Any) -> STTResult:
        """Transcribe audio using the OpenAI Whisper API.

        Args:
            audio: The audio data to transcribe.
            **kwargs: Additional parameters (language, etc.).

        Returns:
            An STTResult with the transcription.
        """
        import soundfile as sf

        client = self._get_client()
        language = kwargs.get("language", "en")

        # Write audio to in-memory WAV buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio.samples, audio.sample_rate, format="WAV")
        buffer.seek(0)
        buffer.name = "audio.wav"

        start = time.perf_counter()
        response = await client.audio.transcriptions.create(
            model=self._model_id,
            file=buffer,
            language=language,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return STTResult(
            text=response.text.strip(),
            latency_ms=latency_ms,
            model_id=self._model_id,
            language=language,
            metadata={"provider": "openai_whisper"},
        )
