from dataclasses import dataclass

import pytest

from app.domain.evidence import Chunk, EvidenceSet, RetrievedChunk
from app.rag.hybrid import HybridRetriever


@dataclass
class StubRetriever:
    results: list[tuple[Chunk, float]]

    def search(self, _query: str, *, top_k: int, min_score: float) -> EvidenceSet:
        return EvidenceSet(
            [
                RetrievedChunk(chunk=chunk, score=score, rank=rank)
                for rank, (chunk, score) in enumerate(self.results, start=1)
                if score >= min_score
            ][:top_k]
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return next((chunk for chunk, _score in self.results if chunk.chunk_id == chunk_id), None)


def chunk(chunk_id: str, text: str = "policy") -> Chunk:
    return Chunk(chunk_id, "doc", "Policy", "v1", "Rule", text)


def test_hybrid_promotes_agreement_and_is_deterministic() -> None:
    shared, lexical_only, semantic_only = chunk("shared"), chunk("lexical"), chunk("semantic")
    retriever = HybridRetriever(
        StubRetriever([(lexical_only, 0.9), (shared, 0.8)]),
        StubRetriever([(semantic_only, 0.95), (shared, 0.7)]),
        lexical_min_score=0.1,
        semantic_min_score=0.6,
    )

    evidence = retriever.search("query", top_k=3, min_score=0.1)

    assert [item.chunk.chunk_id for item in evidence.items] == [
        "shared",
        "lexical",
        "semantic",
    ]
    assert evidence.items[0].score > evidence.items[1].score


def test_hybrid_does_not_force_below_threshold_semantic_candidate() -> None:
    retriever = HybridRetriever(
        StubRetriever([]),
        StubRetriever([(chunk("nearest-but-unsupported"), 0.2)]),
        lexical_min_score=0.1,
        semantic_min_score=0.7,
    )

    assert not retriever.search("unanswerable", top_k=3, min_score=0.1)


def test_optional_semantic_gate_rejects_lexical_false_positive() -> None:
    retriever = HybridRetriever(
        StubRetriever([(chunk("lexical-false-positive"), 0.9)]),
        StubRetriever([(chunk("nearest-but-unsupported"), 0.1)]),
        lexical_min_score=0.1,
        semantic_min_score=0.2,
        semantic_gate=True,
    )

    assert not retriever.search("out of domain", top_k=3, min_score=0.1)


def test_hybrid_rejects_same_id_with_different_content() -> None:
    retriever = HybridRetriever(
        StubRetriever([(chunk("same", "first"), 1.0)]),
        StubRetriever([(chunk("same", "second"), 1.0)]),
        lexical_min_score=0.0,
        semantic_min_score=0.0,
    )

    with pytest.raises(ValueError, match="disagree"):
        retriever.search("query", top_k=2, min_score=0.0)
