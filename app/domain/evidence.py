"""Evidence: the pieces of the knowledge base a run actually saw.

These are plain data structures with no I/O, defined here rather than in
:mod:`app.rag` so that the policy rules which reason about evidence (does the
retrieved set contradict itself? was anything retrieved at all?) stay in the
dependency-free layer and can be table-tested.

The invariant that makes the anti-fabrication check possible is that a run's
:class:`EvidenceSet` is *the* set of chunks the model was shown -- nothing more
and nothing less. The prompt is rendered from it, and the citation validator
resolves against it. If those two ever come from different places, a model
could cite something it was never given and the check would still pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Collapse every run of whitespace to a single space, so a quote that differs
#: from the source only in line wrapping still matches. Anything beyond that --
#: different words, added emphasis, "corrected" grammar -- does not match, on
#: purpose.
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Whitespace-normalised, case-folded form used for quote matching."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable section of one version of one document."""

    chunk_id: str
    document_id: str
    document_title: str
    document_version: str
    heading: str
    text: str
    #: Audience this section applies to, from the document front-matter, e.g.
    #: ``standard`` or ``enterprise``. Two chunks that apply to different
    #: audiences are how "conflicting evidence" is detected deterministically
    #: rather than by asking the model whether it feels conflicted.
    applies_to: str = "all"

    def contains_quote(self, quote: str) -> bool:
        """Whether ``quote`` appears verbatim in this chunk."""
        needle = normalise(quote)
        return bool(needle) and needle in normalise(self.text)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk plus why it was selected."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(slots=True)
class EvidenceSet:
    """Exactly what a run retrieved, in rank order."""

    items: list[RetrievedChunk] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def by_id(self, chunk_id: str) -> Chunk | None:
        for item in self.items:
            if item.chunk.chunk_id == chunk_id:
                return item.chunk
        return None

    @property
    def chunk_ids(self) -> set[str]:
        return {item.chunk.chunk_id for item in self.items}

    @property
    def audiences(self) -> set[str]:
        """Distinct ``applies_to`` values present, excluding the catch-all."""
        return {item.chunk.applies_to for item in self.items if item.chunk.applies_to != "all"}
