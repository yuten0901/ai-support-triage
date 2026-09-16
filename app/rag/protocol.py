"""検索実装が満たす契約。

ワークフローは BM25、dense、hybrid の具体的な保存方式を知りません。すべての
検索器は同じ ``EvidenceSet`` を返し、引用検証に必要な元チャンクも解決できる
必要があります。
"""

from __future__ import annotations

from typing import Protocol

from app.domain.evidence import Chunk, EvidenceSet


class Retriever(Protocol):
    """オーケストレーターが依存する最小の検索契約。"""

    def search(self, query: str, *, top_k: int, min_score: float) -> EvidenceSet:
        """閾値以上の根拠を順位付きで返す。空集合も正しい結果である。"""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """引用検証用に、検索時と同一バージョンのチャンクを解決する。"""
        ...
