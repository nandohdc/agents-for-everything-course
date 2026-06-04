"""CLI entry point to validate document chunking.

Usage::

    python scripts/chunk_documents.py
    python scripts/chunk_documents.py --data-dir path/to/corpus
    python scripts/chunk_documents.py --chunk-size 300 --chunk-overlap 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/chunk_documents.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_document  # noqa: E402
from src.document_loader import load_documents  # noqa: E402

_PREVIEW_CHARS = 120


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and chunk .txt documents from a corpus directory.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the corpus directory (default: data).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Max characters per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Characters shared between consecutive chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    args = parser.parse_args()

    documents = load_documents(args.data_dir)

    total_chunks = 0
    print(
        f"Loaded {len(documents)} document(s); "
        f"chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}\n"
    )
    for doc in documents:
        chunks = chunk_document(doc, args.chunk_size, args.chunk_overlap)
        total_chunks += len(chunks)
        source = doc.metadata.get("path", doc.metadata.get("filename", "<unknown>"))
        print(f"  {source}: {len(chunks)} chunk(s)")
        if chunks:
            preview = chunks[0].text[:_PREVIEW_CHARS]
            suffix = "..." if len(chunks[0].text) > _PREVIEW_CHARS else ""
            print(f"    chunk[0]: {preview}{suffix}")

    print(f"\nTotal: {total_chunks} chunk(s) across {len(documents)} document(s)")


if __name__ == "__main__":
    main()
