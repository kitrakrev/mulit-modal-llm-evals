"""Tests for RAG pipeline components."""

from rag.chunker import chunk_by_separator, chunk_by_tokens
from rag.embeddings import SimpleHashEmbedder
from rag.pipeline import RAGPipeline
from rag.retriever import VectorRetriever, cosine_similarity


class TestChunker:
    """Tests for text chunking functions."""

    def test_chunk_by_tokens(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_by_tokens(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2
        assert chunks[0].index == 0
        assert chunks[1].index == 1
        # First chunk should have 200 tokens
        assert len(chunks[0].text.split()) == 200

    def test_chunk_by_separator(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_by_separator(text, separator="\n\n")
        assert len(chunks) == 3
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."
        assert chunks[2].text == "Third paragraph."

    def test_empty_text(self):
        assert chunk_by_tokens("") == []
        assert chunk_by_tokens("   ") == []
        assert chunk_by_separator("") == []
        assert chunk_by_separator("   ") == []


class TestEmbedder:
    """Tests for the SimpleHashEmbedder."""

    def test_consistent_results(self):
        embedder = SimpleHashEmbedder(dim=128)
        v1 = embedder.embed("hello world")
        v2 = embedder.embed("hello world")
        assert v1 == v2

    def test_correct_dimension(self):
        embedder = SimpleHashEmbedder(dim=64)
        vec = embedder.embed("test text")
        assert len(vec) == 64

    def test_batch_embed(self):
        embedder = SimpleHashEmbedder(dim=128)
        texts = ["hello", "world", "test"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        for vec in results:
            assert len(vec) == 128
        # Each text should produce a different embedding
        assert results[0] != results[1]


class TestRetriever:
    """Tests for the VectorRetriever."""

    def test_retrieve_returns_results(self):
        embedder = SimpleHashEmbedder(dim=128)
        retriever = VectorRetriever(embedder)
        from rag.chunker import Chunk

        chunks = [
            Chunk(text="Python is a programming language", index=0),
            Chunk(text="Java is used for enterprise applications", index=1),
            Chunk(text="Python is great for data science", index=2),
        ]
        retriever.add_chunks(chunks)
        results = retriever.retrieve("Python programming", top_k=2)
        assert len(results) == 2
        assert all(r.score is not None for r in results)
        # Scores should be in descending order
        assert results[0].score >= results[1].score

    def test_empty_retriever(self):
        embedder = SimpleHashEmbedder(dim=128)
        retriever = VectorRetriever(embedder)
        results = retriever.retrieve("test query")
        assert results == []

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec, vec) == 1.0

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9


class TestRAGPipeline:
    """Tests for the RAGPipeline."""

    def test_index_and_retrieve(self):
        pipeline = RAGPipeline(chunk_size=50, chunk_overlap=10, top_k=2)
        docs = [
            {"text": " ".join(["Python programming"] * 30), "source": "doc1"},
            {"text": " ".join(["Java enterprise"] * 30), "source": "doc2"},
        ]
        total = pipeline.index_documents(docs)
        assert total > 0

        result = pipeline.run("Python programming")
        assert result.query == "Python programming"
        assert result.answer != ""
        assert len(result.retrieved_chunks) <= 2

    def test_with_generator_function(self):
        pipeline = RAGPipeline(chunk_size=50, chunk_overlap=10, top_k=2)
        docs = [
            {"text": " ".join(["machine learning"] * 30), "source": "ml_doc"},
        ]
        pipeline.index_documents(docs)

        def fake_generate(query: str, context: str) -> str:
            return f"Answer to '{query}' based on context."

        result = pipeline.run("What is ML?", generate_fn=fake_generate)
        assert result.answer == "Answer to 'What is ML?' based on context."
        assert result.context != ""
