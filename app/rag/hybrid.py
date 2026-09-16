"""語彙検索と意味検索を決定的に統合する rank-fusion 層。"""

from __future__ import annotations

from collections import defaultdict

from app.domain.evidence import Chunk, EvidenceSet, RetrievedChunk
from app.rag.protocol import Retriever


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) で2つの検索器を統合する。

    生スコアは BM25 と cosine similarity で尺度が異なるため加算しない。順位だけを
    統合することで、実装ごとのスコア校正をワークフローへ漏らさず、同順位なら
    ``chunk_id`` で安定化する。

    各検索器にはそれぞれの妥当な下限値を渡す。dense 検索で「必ず最も近い文書」を
    返す設計は未知質問を根拠ありに変えてしまうため禁止する。
    """

    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        *,
        lexical_min_score: float,
        semantic_min_score: float,
        rank_constant: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        self._lexical = lexical
        self._semantic = semantic
        self._lexical_min_score = lexical_min_score
        self._semantic_min_score = semantic_min_score
        self._rank_constant = rank_constant
        self._candidate_multiplier = candidate_multiplier

    def search(self, query: str, *, top_k: int, min_score: float) -> EvidenceSet:
        if top_k < 1:
            return EvidenceSet()
        candidate_count = top_k * self._candidate_multiplier
        result_sets = (
            self._lexical.search(query, top_k=candidate_count, min_score=self._lexical_min_score),
            self._semantic.search(query, top_k=candidate_count, min_score=self._semantic_min_score),
        )

        scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        max_score = 2.0 / (self._rank_constant + 1)
        for evidence in result_sets:
            for item in evidence.items:
                chunk_id = item.chunk.chunk_id
                existing = chunks.get(chunk_id)
                if existing is not None and existing != item.chunk:
                    raise ValueError(f"retrievers disagree on immutable chunk {chunk_id}")
                chunks[chunk_id] = item.chunk
                scores[chunk_id] += 1.0 / (self._rank_constant + item.rank)

        ranked = sorted(
            ((score / max_score, chunk_id) for chunk_id, score in scores.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        items = [
            RetrievedChunk(chunk=chunks[chunk_id], score=round(score, 4), rank=rank)
            for rank, (score, chunk_id) in enumerate(ranked, start=1)
            if score >= min_score
        ][:top_k]
        return EvidenceSet(items=items)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        lexical = self._lexical.get_chunk(chunk_id)
        semantic = self._semantic.get_chunk(chunk_id)
        if lexical is not None and semantic is not None and lexical != semantic:
            raise ValueError(f"retrievers disagree on immutable chunk {chunk_id}")
        return lexical or semantic
