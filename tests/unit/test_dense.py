from pathlib import Path

import httpx
import pytest

from app.rag.dense import DenseIndex
from app.rag.embedding import EmbeddingError, OllamaEmbedder
from app.rag.loader import load_documents


class KeywordEmbedder:
    model_id = "test:keywords"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vocabulary = ("refund", "shipping", "subscription")
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            if lowered == "astronomy":
                vectors.append([0.0, 0.0, 0.0, 1.0])
                continue
            known = [float(lowered.count(term)) for term in vocabulary]
            vectors.append([*known, 0.0] if any(known) else [0.1, 0.1, 0.1, 0.0])
        return vectors


def test_dense_index_retrieves_and_rejects_by_absolute_threshold() -> None:
    index = DenseIndex(load_documents(Path("knowledge")), KeywordEmbedder())

    evidence = index.search("refund refund", top_k=2, min_score=0.8)
    unsupported = index.search("astronomy", top_k=2, min_score=0.9)

    assert evidence
    top = evidence.items[0].chunk
    assert "refund" in f"{top.heading} {top.text}".casefold()
    assert not unsupported


def test_ollama_embedder_uses_batch_endpoint_and_validates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        assert b'"truncate":false' in request.content
        return httpx.Response(
            200,
            json={"model": "embeddinggemma", "embeddings": [[1.0, 0.0], [0.0, 1.0]]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder(base_url="http://ollama.test", client=client)

    assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_ollama_embedder_rejects_wrong_count() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"model": "embeddinggemma", "embeddings": [[1.0]]}
            )
        )
    )
    embedder = OllamaEmbedder(client=client)

    with pytest.raises(EmbeddingError, match="count mismatch"):
        embedder.embed(["first", "second"])
