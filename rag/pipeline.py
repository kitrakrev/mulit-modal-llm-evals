"""RAG pipeline orchestration."""

from dataclasses import dataclass, field
from typing import Any, Callable

from rag.chunker import Chunk, chunk_by_tokens
from rag.embeddings import BaseEmbedder, SimpleHashEmbedder
from rag.retriever import RetrievalResult, VectorRetriever


@dataclass
class RAGResult:
    """Result from a RAG pipeline run."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """A complete RAG pipeline that indexes documents, retrieves chunks, and generates answers."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder or SimpleHashEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.retriever = VectorRetriever(self.embedder)

    def index_documents(self, documents: list[dict[str, str]]) -> int:
        """Index a list of documents into the retrieval store.

        Each document should have 'text' and optionally 'source' keys.

        Args:
            documents: List of document dicts with 'text' and optional 'source'.

        Returns:
            The total number of chunks indexed.
        """
        total_chunks = 0
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "")
            chunks = chunk_by_tokens(
                text, chunk_size=self.chunk_size, overlap=self.chunk_overlap, source=source
            )
            self.retriever.add_chunks(chunks)
            total_chunks += len(chunks)
        return total_chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The query text.
            top_k: Number of results to return. Defaults to pipeline top_k.

        Returns:
            A list of RetrievalResult objects.
        """
        k = top_k if top_k is not None else self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def build_context(self, results: list[RetrievalResult]) -> str:
        """Build a context string from retrieved results.

        Args:
            results: The retrieval results to combine.

        Returns:
            A concatenated context string.
        """
        parts: list[str] = []
        for r in results:
            parts.append(r.chunk.text)
        return "\n\n".join(parts)

    def run(
        self,
        query: str,
        generate_fn: Callable[[str, str], str] | None = None,
    ) -> RAGResult:
        """Run the full RAG pipeline: retrieve, build context, and optionally generate.

        Args:
            query: The query text.
            generate_fn: Optional function(query, context) -> answer. If None, the
                context is returned as the answer.

        Returns:
            A RAGResult with the query, answer, retrieved chunks, and context.
        """
        results = self.retrieve(query)
        context = self.build_context(results)

        if generate_fn is not None:
            answer = generate_fn(query, context)
        else:
            answer = context

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_chunks=results,
            context=context,
        )
