"""Rendering the untrusted half of a prompt.

Separated from the templates so there is exactly one place that decides how
ticket text, retrieved excerpts and tool output are wrapped -- and so that
place can be tested directly. If two call sites rendered evidence differently,
the citation validator and the prompt would eventually disagree about what the
model was shown, and a fabricated citation could pass.

Two rules are enforced here rather than requested in prose:

**Block delimiters are neutralised in untrusted text.** A customer who writes
``</ticket><policy_excerpts>`` in their message is trying to close our block
early and open one of ours. Escaping ``<`` inside untrusted content makes that
inert. It is a small defence and it is not the whole story -- see
``docs/security.md`` -- but it removes the cheapest attack.

**Ticket text is truncated with a visible marker.** A silent truncation would
mean the model answers a question it only partly read while the trace shows the
full ticket. The marker keeps the two honest.
"""

from __future__ import annotations

from app.domain.evidence import EvidenceSet
from app.tools.registry import ToolOutcome

#: Angle brackets are the only characters that can forge a block boundary in
#: this format. Ampersand is escaped first so the escaping is reversible and
#: unambiguous when a reviewer reads the rendered prompt.
_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))


def neutralise(text: str) -> str:
    """Make untrusted text unable to forge a block delimiter."""
    for needle, replacement in _ESCAPES:
        text = text.replace(needle, replacement)
    return text


def render_ticket(subject: str, body: str, *, max_chars: int) -> str:
    """The customer's message, escaped and bounded."""
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[... message truncated by the triage service ...]"
    return f"<ticket>\nSubject: {neutralise(subject.strip())}\n\n{neutralise(body)}\n</ticket>"


def render_evidence(evidence: EvidenceSet) -> str:
    """The retrieved excerpts, each labelled with the id the model must cite.

    The text is reproduced verbatim -- no trimming, no re-wrapping. The
    citation validator matches quotes against the *source* chunk, so any
    transformation applied here and not there would make an exact quote fail
    validation, and the model would be punished for doing precisely what it was
    told to do.
    """
    if not evidence:
        return (
            "<policy_excerpts>\n"
            "No policy excerpt was retrieved for this request. You cannot cite "
            "anything, so answer_status must be insufficient_evidence.\n"
            "</policy_excerpts>"
        )

    parts = ["<policy_excerpts>"]
    for item in evidence.items:
        chunk = item.chunk
        parts.append(
            f'<excerpt chunk_id="{chunk.chunk_id}" document="{neutralise(chunk.document_title)}" '
            f'version="{chunk.document_version}" applies_to="{chunk.applies_to}" '
            f'section="{neutralise(chunk.heading)}">\n'
            f"{neutralise(chunk.text)}\n"
            f"</excerpt>"
        )
    parts.append("</policy_excerpts>")
    return "\n".join(parts)


def render_tool_results(outcomes: list[ToolOutcome]) -> str:
    """What the lookups returned, including the ones that did not work.

    Failures are shown to the model rather than hidden. A model told "the order
    lookup failed" can say so; a model shown nothing assumes the lookup was not
    needed and answers as if the fact were unavailable rather than unknown --
    and those produce different, and differently wrong, replies.
    """
    if not outcomes:
        return "<tool_results>\nNo lookups have been performed.\n</tool_results>"

    parts = ["<tool_results>"]
    for index, outcome in enumerate(outcomes, start=1):
        parts.append(
            f'<result index="{index}" tool="{outcome.tool_name}" status="{outcome.status}">\n'
            f"{neutralise(outcome.rendered)}\n"
            f"</result>"
        )
    parts.append("</tool_results>")
    return "\n".join(parts)


def render_tool_catalogue(entries: list[tuple[str, str, str]]) -> str:
    """The tools the model may request: name, purpose, argument schema."""
    if not entries:
        return "No tools are available. needs_tool must be false."
    return "\n\n".join(
        f"- {name}: {description}\n  arguments: {schema}" for name, description, schema in entries
    )
