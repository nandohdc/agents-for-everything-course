"""Script to build and persist a FAISS index from generated embeddings.

Loads saved embeddings from the indexes/ directory, builds a FAISS index,
and persists the index back to the indexes/ directory.
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/build_index.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.embeddings import load_embeddings  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from saved embeddings."
    )
    parser.add_argument(
        "--input-dir", type=str, default="indexes", help="Directory containing embeddings.npy"
    )
    parser.add_argument(
        "--output-file", type=str, default="indexes/faiss.index", help="Path to save the FAISS index"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not (input_dir / "embeddings.npy").exists():
        print(f"Error: embeddings.npy not found in '{input_dir}'. Please run generate_embeddings.py first.")
        return

    print(f"Loading embeddings from {input_dir}...")
    embeddings, _ = load_embeddings(input_dir)
    
    if embeddings.size == 0:
        print("No embeddings found to index.")
        return
        
    dimension = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {dimension}.")

    print("Building FAISS index...")
    vector_store = VectorStore(dimension=dimension)
    vector_store.add(embeddings)
    
    print(f"Saving FAISS index to {args.output_file}...")
    vector_store.save(args.output_file)
    print("Done!")


if __name__ == "__main__":
    main()
