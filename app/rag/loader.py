"""Turning the knowledge base on disk into retrievable chunks.

The knowledge base is a directory of Markdown files with YAML front matter.
Files, not a database: they diff in review, they version with the code, and
"which policy did the system have on the day it made that decision?" is
answered by ``git log`` rather than by an audit table nobody wrote.

Three properties of the chunking matter more than the chunk size:

**Chunk ids contain the document version.** A citation is ``chunk_id`` plus a
quote, so a citation minted against version ``2026-02-01`` cannot silently
resolve against ``2026-05-01`` after the policy changes -- the id no longer
exists. The failure is loud, at exactly the moment a stale answer would
otherwise have looked fine.

**Sections are the unit.** Splitting on ``##`` boundaries means a chunk is a
whole policy rule with its heading attached, so a quote from it is legible on
its own. Fixed-width windows would cut rules in half and produce citations that
are technically verbatim and practically meaningless.

**Audience is a first-class field.** ``applies_to`` comes from the front matter
and can be overridden per section. It is what lets the system notice
deterministically that it retrieved one rule for standard customers and a
different one for enterprise customers, instead of asking the model whether it
feels conflicted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.domain.evidence import Chunk

#: Sections longer than this are split on paragraph boundaries. Generous,
#: because policy sections are short and splitting one is the exception; the
#: limit exists so a pathological document cannot blow up a prompt.
MAX_CHUNK_CHARS = 1_400

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
#: A section may override the document's audience with a line immediately
#: after its heading, e.g. ``applies_to: enterprise``.
_SECTION_ATTR = re.compile(r"^applies_to:\s*(\S+)\s*$", re.MULTILINE)


class KnowledgeError(Exception):
    """A document could not be loaded. Loud on purpose: a knowledge base that
    silently drops a malformed file answers fewer questions than it should and
    gives no indication why."""


@dataclass(frozen=True, slots=True)
class Document:
    """One knowledge-base file."""

    document_id: str
    title: str
    version: str
    applies_to: str
    path: str
    #: Content hash. Two deployments with the same checksum have the same
    #: knowledge, which is the cheap way to check that a report generated
    #: last week describes the documents in front of you now.
    checksum: str
    chunks: tuple[Chunk, ...]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug or "section"


def _split_long(text: str, limit: int) -> list[str]:
    """Split on blank lines, packing paragraphs up to ``limit``."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        # +2 for the blank line that will rejoin them.
        if current and size + len(paragraph) + 2 > limit:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts or [text.strip()]


def parse_document(path: Path, raw: str) -> Document:
    """Parse one Markdown file into a :class:`Document`."""
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise KnowledgeError(f"{path.name}: missing YAML front matter")

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise KnowledgeError(f"{path.name}: front matter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise KnowledgeError(f"{path.name}: front matter must be a mapping")

    missing = [key for key in ("id", "title", "version") if not meta.get(key)]
    if missing:
        raise KnowledgeError(f"{path.name}: front matter missing {', '.join(missing)}")

    document_id = str(meta["id"])
    title = str(meta["title"])
    version = str(meta["version"])
    default_audience = str(meta.get("applies_to", "all"))

    body = raw[match.end() :]
    checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    chunks = _chunk_body(
        body,
        document_id=document_id,
        title=title,
        version=version,
        default_audience=default_audience,
    )
    if not chunks:
        raise KnowledgeError(f"{path.name}: no '## ' sections found; nothing to retrieve")

    return Document(
        document_id=document_id,
        title=title,
        version=version,
        applies_to=default_audience,
        path=path.name,
        checksum=checksum,
        chunks=tuple(chunks),
    )


def _chunk_body(
    body: str, *, document_id: str, title: str, version: str, default_audience: str
) -> list[Chunk]:
    headings = list(_SECTION.finditer(body))
    chunks: list[Chunk] = []

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        section_text = body[heading.end() : end].strip()
        if not section_text:
            continue

        audience = default_audience
        attr = _SECTION_ATTR.search(section_text)
        if attr is not None and attr.start() == 0:
            audience = attr.group(1)
            section_text = section_text[attr.end() :].strip()

        heading_text = heading.group(1)
        slug = _slugify(heading_text)
        parts = _split_long(section_text, MAX_CHUNK_CHARS)

        for part_index, part in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f"-p{part_index}"
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}@{version}#{slug}{suffix}",
                    document_id=document_id,
                    document_title=title,
                    document_version=version,
                    heading=heading_text,
                    text=part,
                    applies_to=audience,
                )
            )

    return chunks


def load_documents(directory: Path) -> list[Document]:
    """Load every ``*.md`` file in ``directory``, sorted by filename.

    Sorted so the corpus -- and therefore every BM25 score derived from it --
    is identical on every machine. An evaluation report that depends on
    filesystem iteration order is not reproducible.
    """
    if not directory.is_dir():
        raise KnowledgeError(f"knowledge directory not found: {directory}")

    documents = [
        parse_document(path, path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.md"))
    ]
    if not documents:
        raise KnowledgeError(f"no Markdown documents in {directory}")

    seen: dict[str, str] = {}
    for document in documents:
        key = f"{document.document_id}@{document.version}"
        if key in seen:
            raise KnowledgeError(
                f"duplicate document id {key} in {seen[key]} and {document.path}; "
                "chunk ids would collide and citations could resolve to the wrong text"
            )
        seen[key] = document.path

    return documents
