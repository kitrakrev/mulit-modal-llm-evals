"""Custom endpoint provider using OpenAI-compatible API format."""

import os
import time
from typing import Any

from models.providers.base import BaseProvider, ModelResponse


class CustomProvider(BaseProvider):
    """Provider for OpenAI-compatible custom endpoints.

    Connects to any endpoint that follows the OpenAI API format, such as
    vLLM, Ollama, or other self-hosted services.
    """

    def __init__(
        self,
        model_id: str = "custom-model",
        base_url: str = "http://localhost:8080/v1",
        api_key_env: str = "CUSTOM_API_KEY",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._extra = kwargs
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "custom"

    def _get_client(self) -> Any:
        """Get or create the OpenAI-compatible client.

        Returns:
            An AsyncOpenAI client configured with the custom base URL.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = os.environ.get(self._api_key_env, "no-key-required")
            self._client = AsyncOpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using the custom endpoint.

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
                "base_url": self._base_url,
            },
        )

    async def health_check(self) -> bool:
        """Check if the custom endpoint is reachable.

        Returns:
            True if the endpoint responds, False otherwise.
        """
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False
