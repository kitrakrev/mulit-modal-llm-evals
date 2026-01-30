"""Retrieval components for RAG pipelines."""

import math
from dataclasses import dataclass

from rag.chunker import Chunk
from rag.embeddings import BaseEmbedder


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorRetriever:
    """A simple in-memory vector retriever."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the retriever index.

        Args:
            chunks: The chunks to index.
        """
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for a query.

        Args:
            query: The query text.
            top_k: Number of results to return.

        Returns:
            A list of RetrievalResult objects sorted by descending score.
        """
        if not self._chunks:
            return []

        query_embedding = self.embedder.embed(query)
        scored: list[RetrievalResult] = []

        for chunk, emb in zip(self._chunks, self._embeddings):
            score = cosine_similarity(query_embedding, emb)
            scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
