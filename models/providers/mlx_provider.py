"""MLX local model provider implementation."""

import time
from typing import Any

from models.providers.base import BaseProvider, ModelResponse


class MLXProvider(BaseProvider):
    """Provider that uses mlx_lm for local inference on Apple Silicon.

    Handles model loading with caching so that repeated calls avoid
    reloading the model from disk.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._extra = kwargs
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def provider_name(self) -> str:
        return "mlx"

    def _load_model(self) -> tuple[Any, Any]:
        """Load the model and tokenizer, caching for reuse.

        Returns:
            A tuple of (model, tokenizer).
        """
        if self._model is None or self._tokenizer is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self._model_id)
        return self._model, self._tokenizer

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response using the local MLX model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters (max_tokens, temperature, etc.).

        Returns:
            A ModelResponse with the generated text and metadata.
        """
        from mlx_lm import generate as mlx_generate

        model, tokenizer = self._load_model()

        max_tokens = kwargs.get("max_tokens", self._extra.get("max_tokens", 512))
        temperature = kwargs.get("temperature", self._extra.get("temperature", 0.7))

        # Apply chat template if the tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        start = time.perf_counter()
        response_text = mlx_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Approximate token counts from text
        input_tokens = len(tokenizer.encode(formatted_prompt))
        output_tokens = len(tokenizer.encode(response_text))

        return ModelResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self._model_id,
            metadata={
                "provider": self.provider_name,
                "quantization": self._extra.get("quantization", "unknown"),
            },
        )

    async def health_check(self) -> bool:
        """Check if the MLX model can be loaded.

        Returns:
            True if mlx_lm is available, False otherwise.
        """
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False
