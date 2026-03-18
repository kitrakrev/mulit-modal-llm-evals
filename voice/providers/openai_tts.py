"""OpenAI TTS provider for cloud text-to-speech."""

import os
import time
from typing import Any

import numpy as np

from voice.tts_runner import TTSResult


class OpenAITTSProvider:
    """Text-to-speech provider using the OpenAI TTS API.

    Supports multiple voices and output formats via the OpenAI audio speech endpoint.
    """

    def __init__(
        self,
        model_id: str = "tts-1",
        voice: str = "alloy",
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._voice = voice
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

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        """Synthesize speech from text using the OpenAI TTS API.

        Args:
            text: The input text to synthesize.
            **kwargs: Additional parameters (voice, speed, response_format, etc.).

        Returns:
            A TTSResult with the synthesized audio.
        """
        import io

        import soundfile as sf

        client = self._get_client()
        voice = kwargs.get("voice", self._voice)
        speed = kwargs.get("speed", 1.0)

        start = time.perf_counter()
        response = await client.audio.speech.create(
            model=self._model_id,
            voice=voice,
            input=text,
            speed=speed,
            response_format="wav",
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Read audio from response bytes
        audio_bytes = response.content
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer)

        return TTSResult(
            audio_data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            latency_ms=latency_ms,
            model_id=self._model_id,
            metadata={
                "provider": "openai_tts",
                "voice": voice,
                "speed": speed,
            },
        )
