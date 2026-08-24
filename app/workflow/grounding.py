"""The check that makes a citation mean something.

Most "RAG with citations" systems are citation-shaped rather than cited: the
model is handed some documents and asked to name its sources, and whatever it
names is displayed. Nothing verifies that the named source exists, and nothing
verifies that it says what the answer claims. Both failures look exactly like
success in the response body.

This module closes both, and neither check is a matter of degree:

**The source must be one the model was actually shown.** Chunk ids are resolved
against the run's own :class:`EvidenceSet` -- the same object the prompt was
rendered from. A model cannot cite a document it merely remembers, because the
id it would have to produce is not in the set.

**The quote must appear in that source, verbatim.** Whitespace is normalised
and case is folded, because line wrapping is not a meaning change. Everything
else -- a reworded phrase, a "corrected" sentence, two half-sentences joined --
fails. That is the point: a model that cites a real document and then
paraphrases it into saying something it does not say has fabricated a fact
while appearing to be scrupulous, and it is the hardest fabrication to spot by
eye.

The consistency rules below matter for the same reason. An answer with no
citations is not an answer; a non-answer that arrives with a helpful draft
attached will have that draft sent to a customer by some future caller who
assumed the draft field was only populated when it was safe to use.

A failure here is repairable once -- the model is told exactly what did not
resolve and given the option of admitting the evidence is not there. If it
fails again the run ends as ``UNGROUNDED_CITATION``, which is a *system* error:
it means either the prompt or the model has regressed, and someone should look
at it.
"""

from __future__ import annotations

from app.ai.schemas import Resolution
from app.domain.evidence import EvidenceSet

#: Undo the escaping applied when the excerpt was rendered into the prompt.
#: Without this, a policy document containing an angle bracket would make every
#: correct quote from it fail -- the model would be quoting exactly what it was
#: shown, and being told it had fabricated. The direction is safe: unescaping
#: can only turn the model's text into something closer to the source, and the
#: source itself is never transformed.
_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"))


def _unescape(text: str) -> str:
    for needle, replacement in _ENTITIES:
        text = text.replace(needle, replacement)
    return text


def validate_resolution(resolution: Resolution, evidence: EvidenceSet) -> str | None:
    """Return why this resolution is not acceptable, or ``None`` if it is.

    The string is written to be handed straight back to the model, so it names
    the specific id or quote that failed rather than saying "invalid".
    """
    problems: list[str] = []

    if resolution.answer_status == "answered":
        if not resolution.citations:
            problems.append(
                "answer_status is 'answered' but citations is empty. An answer must cite "
                "at least one excerpt, or answer_status must be 'insufficient_evidence'."
            )
        if not resolution.reply_draft.strip():
            problems.append("answer_status is 'answered' but reply_draft is empty.")
    else:
        if resolution.citations:
            problems.append(
                f"answer_status is '{resolution.answer_status}' but {len(resolution.citations)} "
                "citation(s) were given. A non-answer cites nothing."
            )
        if resolution.reply_draft.strip():
            problems.append(
                f"answer_status is '{resolution.answer_status}' but reply_draft is not empty. "
                "Leave it empty; there is no answer to send."
            )
        if resolution.recommended_action.kind.value != "none":
            problems.append(
                f"answer_status is '{resolution.answer_status}' but recommended_action.kind is "
                f"'{resolution.recommended_action.kind.value}'. It must be 'none'."
            )

    for index, citation in enumerate(resolution.citations, start=1):
        chunk = evidence.by_id(citation.chunk_id)
        if chunk is None:
            available = ", ".join(sorted(evidence.chunk_ids)) or "(none were retrieved)"
            problems.append(
                f"citation {index} names chunk_id '{citation.chunk_id}', which was not among the "
                f"excerpts you were given. Available chunk_ids: {available}."
            )
            continue
        if not chunk.contains_quote(_unescape(citation.quote)):
            problems.append(
                f"citation {index} quotes text that does not appear in '{citation.chunk_id}'. "
                "The quote must be copied character-for-character from that excerpt."
            )

    action = resolution.recommended_action
    if action.kind.value == "issue_refund":
        if action.amount_minor is None or action.amount_minor <= 0:
            problems.append(
                "recommended_action.kind is 'issue_refund' but amount_minor is missing or not "
                "positive. A refund needs an amount."
            )
        if not action.target_id:
            problems.append(
                "recommended_action.kind is 'issue_refund' but target_id is missing. A refund "
                "needs the order it applies to."
            )

    return " ".join(problems) if problems else None


def mark_cited(resolution: Resolution, evidence: EvidenceSet) -> dict[str, str]:
    """Which retrieved chunk ids the accepted resolution actually quoted.

    Called only after validation has passed, so every id here resolves. The
    result feeds ``EvidenceRecord.was_cited``, which is what makes "retrieved
    four, used one" visible in the trace instead of the two being conflated.
    """
    return {
        citation.chunk_id: citation.quote
        for citation in resolution.citations
        if citation.chunk_id in evidence.chunk_ids
    }
