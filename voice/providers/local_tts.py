"""Local TTS provider using pyttsx3 or other local engines."""

import time
from typing import Any

import numpy as np

from voice.tts_runner import TTSResult


class LocalTTSProvider:
    """Text-to-speech provider using local synthesis engines.

    Supports pyttsx3 for offline TTS on macOS, Windows, and Linux.
    Falls back to a simple sine-wave placeholder when no engine is available.
    """

    def __init__(
        self,
        model_id: str = "local-tts",
        sample_rate: int = 22050,
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._extra = kwargs
        self._engine: Any = None

    def _get_engine(self) -> Any | None:
        """Get or create the local TTS engine.

        Returns:
            A pyttsx3 engine or None if not available.
        """
        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
            except (ImportError, Exception):
                return None
        return self._engine

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        """Synthesize speech from text using the local TTS engine.

        Args:
            text: The input text to synthesize.
            **kwargs: Additional parameters (rate, volume, etc.).

        Returns:
            A TTSResult with the synthesized audio.
        """
        start = time.perf_counter()

        engine = self._get_engine()
        if engine is not None:
            audio_data = self._synthesize_with_engine(engine, text, **kwargs)
        else:
            # Placeholder: generate silence with duration proportional to text length
            duration = max(0.5, len(text.split()) * 0.3)
            num_samples = int(duration * self._sample_rate)
            audio_data = np.zeros(num_samples, dtype=np.float32)

        latency_ms = (time.perf_counter() - start) * 1000.0

        return TTSResult(
            audio_data=audio_data,
            sample_rate=self._sample_rate,
            latency_ms=latency_ms,
            model_id=self._model_id,
            metadata={
                "provider": "local_tts",
                "engine_available": engine is not None,
            },
        )

    def _synthesize_with_engine(
        self, engine: Any, text: str, **kwargs: Any
    ) -> np.ndarray:
        """Synthesize using pyttsx3 engine to a temporary file and read back.

        Args:
            engine: The pyttsx3 engine instance.
            text: The input text.
            **kwargs: Additional pyttsx3 parameters.

        Returns:
            Audio samples as numpy array.
        """
        import tempfile
        from pathlib import Path

        import soundfile as sf

        rate = kwargs.get("rate", 150)
        engine.setProperty("rate", rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                data, _sr = sf.read(tmp_path)
                return data.astype(np.float32)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Fallback if engine didn't produce output
        duration = max(0.5, len(text.split()) * 0.3)
        return np.zeros(int(duration * self._sample_rate), dtype=np.float32)
