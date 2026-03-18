"""OpenAI model provider implementation."""

import os
import time
from typing import Any

from models.providers.base import BaseProvider, ModelResponse


class OpenAIProvider(BaseProvider):
    """Provider that uses the OpenAI SDK for chat completions."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._extra = kwargs
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        """Get or create the OpenAI client.

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

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using OpenAI chat completions.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            A ModelResponse with the generated text and metadata.
        """
        client = self._get_client()

        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        model = kwargs.pop("model", self._model_id)
        temperature = kwargs.pop("temperature", self._extra.get("temperature", 0.7))
        max_tokens = kwargs.pop("max_tokens", self._extra.get("max_tokens", 1024))

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_id=model,
            metadata={
                "finish_reason": choice.finish_reason,
                "provider": self.provider_name,
            },
        )

    async def health_check(self) -> bool:
        """Check if the OpenAI API is reachable.

        Returns:
            True if the API key is configured, False otherwise.
        """
        return bool(os.environ.get(self._api_key_env))
