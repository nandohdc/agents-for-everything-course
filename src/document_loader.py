"""Document loader for the local text corpus.

Scans a directory for .txt files and returns their contents with metadata,
ready for downstream embedding and indexing in the RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Document:
    """An immutable document loaded from the corpus.

    Attributes:
        text: The full text content of the document.
        metadata: A dictionary with at least ``filename`` and ``path`` keys.
    """

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def load_documents(data_dir: str | Path = "data") -> List[Document]:
    """Load all ``.txt`` files from *data_dir* recursively.

    Parameters
    ----------
    data_dir:
        Path to the corpus directory.  Defaults to ``"data"``.

    Returns
    -------
    list[Document]
        Documents sorted by their relative POSIX path for deterministic
        ordering.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    NotADirectoryError
        If *data_dir* exists but is not a directory.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {data_path}")

    documents: list[Document] = []

    for txt_file in sorted(data_path.rglob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        relative_path = txt_file.relative_to(data_path.parent)
        metadata = {
            "filename": txt_file.name,
            "path": relative_path.as_posix(),
        }
        documents.append(Document(text=text, metadata=metadata))

    return documents
