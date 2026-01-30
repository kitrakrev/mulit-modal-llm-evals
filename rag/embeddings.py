"""Embedding interfaces and implementations for RAG pipelines."""

import hashlib
import math
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        ...


class SimpleHashEmbedder(BaseEmbedder):
    """A deterministic hash-based embedder for testing purposes.

    Uses SHA256 to generate consistent float vectors from text input.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text using SHA256 hashing.

        Args:
            text: The text to embed.

        Returns:
            A normalized vector of floats with the configured dimension.
        """
        raw: list[float] = []
        # Generate enough hash bytes to fill the dimension
        rounds = math.ceil(self.dim * 4 / 32)  # 4 bytes per float, 32 bytes per SHA256
        for i in range(rounds):
            h = hashlib.sha256(f"{text}:{i}".encode()).digest()
            for j in range(0, len(h), 4):
                if len(raw) >= self.dim:
                    break
                # Convert 4 bytes to a float in [-1, 1]
                val = int.from_bytes(h[j : j + 4], "big") / (2**32) * 2 - 1
                raw.append(val)

        raw = raw[: self.dim]

        # Normalize the vector
        magnitude = math.sqrt(sum(x * x for x in raw))
        if magnitude > 0:
            raw = [x / magnitude for x in raw]

        return raw

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors.
        """
        return [self.embed(t) for t in texts]
