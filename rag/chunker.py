"""Text chunking utilities for RAG pipelines."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text from a document."""

    text: str
    index: int
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_by_tokens(
    text: str, chunk_size: int = 200, overlap: int = 50, source: str = ""
) -> list[Chunk]:
    """Split text into chunks by token count (approximated by whitespace splitting).

    Args:
        text: The text to chunk.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.
        source: Source identifier for the chunks.

    Returns:
        A list of Chunk objects.
    """
    if not text.strip():
        return []

    tokens = text.split()
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(Chunk(text=chunk_text, index=index, source=source))
        index += 1
        start += chunk_size - overlap
        if start >= len(tokens):
            break

    return chunks


def chunk_by_separator(
    text: str, separator: str = "\n\n", source: str = ""
) -> list[Chunk]:
    """Split text into chunks using a separator string.

    Args:
        text: The text to chunk.
        separator: The separator to split on.
        source: Source identifier for the chunks.

    Returns:
        A list of Chunk objects.
    """
    if not text.strip():
        return []

    parts = text.split(separator)
    chunks: list[Chunk] = []

    for index, part in enumerate(parts):
        stripped = part.strip()
        if stripped:
            chunks.append(Chunk(text=stripped, index=index, source=source))

    return chunks
