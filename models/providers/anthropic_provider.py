"""Anthropic model provider implementation."""

import os
import time
from typing import Any

from models.providers.base import BaseProvider, ModelResponse


class AnthropicProvider(BaseProvider):
    """Provider that uses the Anthropic SDK for message generation."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key_env: str = "ANTHROPIC_API_KEY",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._api_key_env = api_key_env
        self._extra = kwargs
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        """Get or create the Anthropic client.

        Returns:
            An AsyncAnthropic client instance.

        Raises:
            RuntimeError: If the API key environment variable is not set.
        """
        if self._client is None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable '{self._api_key_env}' is not set."
                )
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using Anthropic messages API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters (temperature, max_tokens, system, etc.).

        Returns:
            A ModelResponse with the generated text and metadata.
        """
        client = self._get_client()

        model = kwargs.pop("model", self._model_id)
        max_tokens = kwargs.pop("max_tokens", self._extra.get("max_tokens", 1024))
        temperature = kwargs.pop("temperature", self._extra.get("temperature", 0.7))
        system = kwargs.pop("system", None)
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            create_kwargs["system"] = system

        start = time.perf_counter()
        response = await client.messages.create(**create_kwargs, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return ModelResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model_id=model,
            metadata={
                "stop_reason": response.stop_reason,
                "provider": self.provider_name,
            },
        )

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        Returns:
            True if the API key is configured, False otherwise.
        """
        return bool(os.environ.get(self._api_key_env))
