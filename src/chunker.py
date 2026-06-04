"""Text chunking and preprocessing for the RAG pipeline.

Splits loaded :class:`~src.document_loader.Document` objects into smaller,
retrievable :class:`Chunk` objects using a character-based sliding window.
Chunking happens *after* whitespace normalization so that downstream embedding
and indexing see clean, consistently sized text.

Chunk size and overlap are measured in **characters** (not tokens) to keep this
module dependency-free, in line with the project's simple-RAG scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.document_loader import Document

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """An immutable slice of a source document.

    Attributes:
        text: The normalized chunk text.
        metadata: A dictionary with the keys ``source_filename``,
            ``source_path``, ``chunk_index`` (0-based position within the
            source document), and ``chunk_size`` (character length of
            ``text``).
    """

    text: str
    metadata: dict = field(default_factory=dict)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a single :class:`Document` into overlapping :class:`Chunk` objects.

    The document text is whitespace-normalized first, then sliced with a
    sliding window that advances by ``chunk_size - chunk_overlap`` characters.
    Empty chunks (which only occur for blank/whitespace-only documents) are
    discarded, so a document with no textual content yields an empty list.

    Parameters
    ----------
    doc:
        The source document. Its ``filename`` and ``path`` metadata are copied
        onto every resulting chunk.
    chunk_size:
        Maximum number of characters per chunk. Must be ``> 0``.
    chunk_overlap:
        Number of characters shared between consecutive chunks. Must satisfy
        ``0 <= chunk_overlap < chunk_size``.

    Returns
    -------
    list[Chunk]
        Chunks in document order, with sequential ``chunk_index`` values.

    Raises
    ------
    ValueError
        If ``chunk_size`` or ``chunk_overlap`` are out of range.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size "
            f"(got chunk_overlap={chunk_overlap}, chunk_size={chunk_size})"
        )

    text = _normalize_whitespace(doc.text)
    if not text:
        return []

    source_filename = doc.metadata.get("filename", "")
    source_path = doc.metadata.get("path", "")
    step = chunk_size - chunk_overlap

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk_text = text[start:end]
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "source_filename": source_filename,
                        "source_path": source_path,
                        "chunk_index": index,
                        "chunk_size": len(chunk_text),
                    },
                )
            )
            index += 1
        if end == length:
            break
        start += step

    return chunks


def chunk_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Apply :func:`chunk_document` to every document in *docs*.

    Returns the concatenated chunks in input order. ``chunk_index`` restarts at
    0 for each source document, so identify a chunk by ``(source_path,
    chunk_index)``.
    """

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks
