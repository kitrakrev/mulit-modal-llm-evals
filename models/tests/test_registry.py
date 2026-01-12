"""Tests for model registry."""

import tempfile
from pathlib import Path

import pytest
import yaml

from models.registry import ModelConfig, ModelRegistry


class TestModelConfig:
    """Tests for ModelConfig pydantic model."""

    def test_basic_config(self):
        config = ModelConfig(provider="openai", model_id="gpt-4o", type=["text"])
        assert config.provider == "openai"
        assert config.model_id == "gpt-4o"
        assert config.type == ["text"]
        assert config.base_url is None
        assert config.api_key_env is None
        assert config.extra == {}

    def test_full_config(self):
        config = ModelConfig(
            provider="custom",
            model_id="my-model",
            type=["text", "rag"],
            base_url="http://localhost:8080",
            api_key_env="MY_API_KEY",
            extra={"temperature": 0.7},
        )
        assert config.base_url == "http://localhost:8080"
        assert config.api_key_env == "MY_API_KEY"
        assert config.extra["temperature"] == 0.7


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get_model(self):
        registry = ModelRegistry()
        config = ModelConfig(provider="openai", model_id="gpt-4o", type=["text"])
        registry.register_model("gpt-4o", config)
        retrieved = registry.get_model("gpt-4o")
        assert retrieved.model_id == "gpt-4o"

    def test_nonexistent_model_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_model("nonexistent")

    def test_list_with_type_filter(self):
        registry = ModelRegistry()
        registry.register_model(
            "model-a", ModelConfig(provider="p", model_id="a", type=["text", "rag"])
        )
        registry.register_model(
            "model-b", ModelConfig(provider="p", model_id="b", type=["text"])
        )
        registry.register_model(
            "model-c", ModelConfig(provider="p", model_id="c", type=["voice"])
        )

        text_models = registry.list_models(type_filter="text")
        assert "model-a" in text_models
        assert "model-b" in text_models
        assert "model-c" not in text_models

        rag_models = registry.list_models(type_filter="rag")
        assert rag_models == ["model-a"]

        all_models = registry.list_models()
        assert len(all_models) == 3

    def test_load_from_yaml(self):
        models_data = {
            "models": {
                "gpt-4o": {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "type": ["text"],
                    "api_key_env": "OPENAI_API_KEY",
                },
                "local-llm": {
                    "provider": "mlx",
                    "model_id": "llama-3.2-1b",
                    "type": ["text"],
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(models_data, f)
            tmp_path = f.name

        registry = ModelRegistry()
        registry.load_from_yaml(tmp_path)

        model = registry.get_model("gpt-4o")
        assert model.provider == "openai"
        assert model.api_key_env == "OPENAI_API_KEY"

        local = registry.get_model("local-llm")
        assert local.provider == "mlx"

        Path(tmp_path).unlink()

    def test_nonexistent_file_raises(self):
        registry = ModelRegistry()
        with pytest.raises(FileNotFoundError):
            registry.load_from_yaml("/nonexistent/path/models.yaml")
