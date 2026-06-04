"""Script to query the FAISS index and return relevant chunks.

Queries the loaded vector store and metadata for a given text.
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/query.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.query_engine import QueryEngine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Query the FAISS index and return relevant chunks."
    )
    parser.add_argument(
        "query", type=str, help="The query string to search for"
    )
    parser.add_argument(
        "--index-file", type=str, default="indexes/faiss.index", help="Path to the FAISS index"
    )
    parser.add_argument(
        "--metadata-dir", type=str, default="indexes", help="Directory containing metadata.json"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="Number of chunks to retrieve"
    )

    args = parser.parse_args()

    index_path = Path(args.index_file)
    metadata_dir = Path(args.metadata_dir)

    if not index_path.exists():
        print(f"Error: index file not found at '{index_path}'. Please run build_index.py first.")
        return
        
    if not (metadata_dir / "metadata.json").exists():
        print(f"Error: metadata.json not found in '{metadata_dir}'. Please run generate_embeddings.py first.")
        return

    print("Initializing QueryEngine...")
    engine = QueryEngine(
        index_path=index_path,
        metadata_dir=metadata_dir
    )

    print(f"\nQuerying for: '{args.query}' (top {args.top_k} results)\n")
    results = engine.query(args.query, k=args.top_k)
    
    if not results:
        print("No results found.")
        return

    for i, (metadata, score) in enumerate(results, 1):
        print(f"--- Result {i} (Score: {score:.4f}) ---")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Chunk Index: {metadata.get('chunk_index', 'Unknown')}")
        print("Text:")
        print(metadata.get('text', '').strip())
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
