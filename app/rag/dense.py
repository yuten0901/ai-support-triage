"""小規模ベンチマーク用のインメモリ cosine 検索。"""

from __future__ import annotations

import math

from app.domain.evidence import Chunk, EvidenceSet, RetrievedChunk
from app.rag.embedding import Embedder, EmbeddingError
from app.rag.loader import Document


def _normalise(vector: list[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0 or not math.isfinite(magnitude):
        raise EmbeddingError("embedding vector must have a finite, non-zero magnitude")
    return tuple(value / magnitude for value in vector)


class DenseIndex:
    """文書と問い合わせを同一モデルで埋め込み、cosine 類似度で検索する。

    これは比較実験と小規模デモ用であり、永続化や複数プロセス共有を主張しない。
    大規模モードでは同じ ``Retriever`` 契約を満たす pgvector 実装へ差し替える。
    """

    def __init__(self, documents: list[Document], embedder: Embedder) -> None:
        self._chunks = [chunk for document in documents for chunk in document.chunks]
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._embedder = embedder
        inputs = [f"{chunk.heading}\n{chunk.text}" for chunk in self._chunks]
        vectors = embedder.embed_documents(inputs)
        if len(vectors) != len(self._chunks):
            raise EmbeddingError(
                f"embedding count mismatch: expected {len(self._chunks)}, got {len(vectors)}"
            )
        self._vectors = [_normalise(vector) for vector in vectors]
        dimensions = {len(vector) for vector in self._vectors}
        if len(dimensions) > 1:
            raise EmbeddingError("document embeddings have inconsistent dimensions")
        self._dimensions = next(iter(dimensions), 0)

    def search(self, query: str, *, top_k: int, min_score: float) -> EvidenceSet:
        if not query.strip() or top_k < 1 or not self._chunks:
            return EvidenceSet()
        query_vector = _normalise(self._embedder.embed_query(query))
        if len(query_vector) != self._dimensions:
            raise EmbeddingError(
                f"query embedding dimension {len(query_vector)} does not match index "
                f"dimension {self._dimensions}"
            )

        scored = [
            (sum(left * right for left, right in zip(query_vector, vector, strict=True)), chunk)
            for vector, chunk in zip(self._vectors, self._chunks, strict=True)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        items = [
            RetrievedChunk(chunk=chunk, score=round(score, 4), rank=rank)
            for rank, (score, chunk) in enumerate(scored, start=1)
            if score >= min_score
        ][:top_k]
        return EvidenceSet(items=items)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)
