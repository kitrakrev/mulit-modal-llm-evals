"""Model registry for managing model configurations and providers."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from models.providers.base import BaseProvider


class ModelConfig(BaseModel):
    """Configuration for a registered model."""

    provider: str
    model_id: str
    type: list[str] = Field(default_factory=list)
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    """Registry for managing model configurations and providers."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        self._providers: dict[str, BaseProvider] = {}

    def load_from_yaml(self, path: str | Path) -> None:
        """Load model configurations from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        models = data.get("models", {})
        for model_key, model_data in models.items():
            config = ModelConfig(**model_data)
            self._models[model_key] = config

    def register_model(self, name: str, config: ModelConfig) -> None:
        """Register a model configuration.

        Args:
            name: The name to register the model under.
            config: The model configuration.
        """
        self._models[name] = config

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register a model provider.

        Args:
            name: The name to register the provider under.
            provider: The provider instance.
        """
        self._providers[name] = provider

    def get_model(self, name: str) -> ModelConfig:
        """Get a model configuration by name.

        Args:
            name: The model name.

        Returns:
            The model configuration.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered.")
        return self._models[name]

    def get_provider(self, name: str) -> BaseProvider:
        """Get a provider by name.

        Args:
            name: The provider name.

        Returns:
            The provider instance.

        Raises:
            KeyError: If the provider is not registered.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' is not registered.")
        return self._providers[name]

    def list_models(self, type_filter: str | None = None) -> list[str]:
        """List registered model names, optionally filtered by type.

        Args:
            type_filter: If provided, only return models that include this type.

        Returns:
            A list of model names.
        """
        if type_filter is None:
            return list(self._models.keys())
        return [
            name
            for name, config in self._models.items()
            if type_filter in config.type
        ]

    def list_providers(self) -> list[str]:
        """List registered provider names.

        Returns:
            A list of provider names.
        """
        return list(self._providers.keys())
