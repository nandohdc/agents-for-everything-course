"""Script to generate embeddings for all text chunks in the corpus.

Loads documents, chunks them, generates embeddings using all-MiniLM-L6-v2,
and persists the output to the indexes/ directory.
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/generate_embeddings.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.chunker import chunk_documents  # noqa: E402
from src.document_loader import load_documents  # noqa: E402
from src.embeddings import Embedder, chunk_to_metadata, save_embeddings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for the local corpus."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing text documents"
    )
    parser.add_argument(
        "--output-dir", type=str, default="indexes", help="Directory to save embeddings"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in characters"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    print(f"Loading documents from {data_dir}...")
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found to process.")
        return

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(f"Created {len(chunks)} chunks.")

    print(
        "Loading embedding model and generating embeddings (this may take a moment)..."
    )
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    print(
        f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}."
    )

    print(f"Saving embeddings and metadata to {args.output_dir}...")
    metadata = [chunk_to_metadata(chunk) for chunk in chunks]
    save_embeddings(embeddings, metadata, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
