"""Retrieval: BM25 over a few dozen chunks.

A vector database would be technology theatre here. The corpus is a handful of
policy documents that fit in memory, the queries are support tickets that share
vocabulary with the policies, and BM25 has one property an embedding index does
not: **the score is explainable**. When retrieval returns the wrong section, you
can point at the terms that matched. That matters more for a system whose whole
argument is "you can tell why it did that" than any recall improvement would.

The score normalisation is the part worth reading. A raw BM25 score is
unbounded and query-dependent, so a fixed cutoff on it is meaningless. The
obvious fix -- divide by the best score in the result set -- is worse than
useless: it forces the top hit to 1.0 for *every* query, including a query
about something the knowledge base has never heard of. The threshold would then
never reject anything, and "insufficient evidence" would become unreachable.

Instead each score is divided by the maximum score the query *could* achieve
(every term matching with saturated term frequency). The result answers "how
much of this query does this chunk actually cover?", stays comparable across
queries, and lets a genuinely unrelated question score near zero and be
dropped -- which is what makes an honest INSUFFICIENT_EVIDENCE possible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.domain.evidence import Chunk, EvidenceSet, RetrievedChunk
from app.rag.loader import Document

_K1 = 1.5
_B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")

#: Deliberately short. An aggressive stopword list removes terms that carry
#: real signal in support text ("not", "can", "no"), and BM25's IDF already
#: discounts anything that appears in most documents.
_STOPWORDS = frozenset(
    """
    a an the and or of to in for on at by is are was were be been being it its this that
    these those with as from we you i my your our their he she they them us if then than
    so but do does did have has had will would can could should may might must about into
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed, singles dropped."""
    return [
        token
        for token in _TOKEN.findall(text.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    ]


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """What the knowledge-sources endpoint exposes about one document."""

    document_id: str
    title: str
    version: str
    applies_to: str
    path: str
    checksum: str
    chunk_count: int


class KnowledgeIndex:
    """An in-memory BM25 index over the knowledge base.

    Immutable once built. Reindexing constructs a new instance and swaps it in,
    so a request in flight keeps the corpus it started with -- a run whose
    citations were validated against one version of a document must not be
    checked against another halfway through.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._chunks: list[Chunk] = [c for d in documents for c in d.chunks]
        self._tokens: list[Counter[str]] = []
        self._lengths: list[int] = []
        doc_frequency: Counter[str] = Counter()

        for chunk in self._chunks:
            # The heading is indexed with the body: section titles carry the
            # vocabulary a customer is most likely to use ("refund window").
            tokens = tokenize(f"{chunk.heading} {chunk.text}")
            counts = Counter(tokens)
            self._tokens.append(counts)
            self._lengths.append(len(tokens))
            doc_frequency.update(counts.keys())

        self._n = len(self._chunks)
        self._avg_length = (sum(self._lengths) / self._n) if self._n else 0.0
        self._idf = {
            term: math.log(1 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_frequency.items()
        }

    # -- introspection ----------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return self._n

    @property
    def documents(self) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_id=d.document_id,
                title=d.title,
                version=d.version,
                applies_to=d.applies_to,
                path=d.path,
                checksum=d.checksum,
                chunk_count=len(d.chunks),
            )
            for d in self._documents
        ]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    # -- retrieval --------------------------------------------------------

    def search(self, query: str, *, top_k: int, min_score: float) -> EvidenceSet:
        """Return the best-scoring chunks above ``min_score``.

        An empty result is a correct answer and is not padded out to ``top_k``.
        Returning the least-bad chunk for an unanswerable question is exactly
        how a system ends up citing a shipping policy at a billing question.
        """
        terms = tokenize(query)
        if not terms or self._n == 0:
            return EvidenceSet()

        # See module docstring: the ceiling a chunk could score on this query.
        ceiling = sum(self._idf.get(term, 0.0) for term in set(terms)) * (_K1 + 1.0)
        if ceiling <= 0.0:
            # Every query term is unknown to the corpus. Nothing can match, and
            # dividing by the ceiling would be a division by zero.
            return EvidenceSet()

        scored: list[tuple[float, int]] = []
        for index in range(self._n):
            raw = self._score(terms, index)
            if raw > 0.0:
                scored.append((raw / ceiling, index))

        # Sort by score, then chunk id: two chunks with identical scores must
        # come back in the same order on every run, or the report is not
        # reproducible.
        scored.sort(key=lambda pair: (-pair[0], self._chunks[pair[1]].chunk_id))

        items: list[RetrievedChunk] = []
        for normalised, index in scored[:top_k]:
            if normalised < min_score:
                break
            items.append(
                RetrievedChunk(
                    chunk=self._chunks[index], score=round(normalised, 4), rank=len(items) + 1
                )
            )
        return EvidenceSet(items=items)

    def _score(self, terms: list[str], index: int) -> float:
        counts = self._tokens[index]
        length = self._lengths[index]
        norm = _K1 * (1 - _B + _B * (length / self._avg_length if self._avg_length else 1.0))
        total = 0.0
        for term in set(terms):
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (frequency * (_K1 + 1.0)) / (frequency + norm)
        return total
