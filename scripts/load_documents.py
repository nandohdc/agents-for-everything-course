"""CLI entry point to validate corpus loading.

Usage::

    python scripts/load_documents.py
    python scripts/load_documents.py --data-dir path/to/corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/load_documents.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.document_loader import load_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load .txt documents from a corpus directory.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the corpus directory (default: data).",
    )
    args = parser.parse_args()

    documents = load_documents(args.data_dir)

    print(f"Loaded {len(documents)} document(s)\n")
    for doc in documents:
        print(
            f"  path={doc.metadata['path']}  "
            f"filename={doc.metadata['filename']}  "
            f"chars={len(doc.text)}"
        )


if __name__ == "__main__":
    main()
