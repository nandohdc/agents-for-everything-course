"""Query engine for retrieving relevant chunks for a user question."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.embeddings import Embedder, load_embeddings
from src.vector_store import VectorStore


class QueryEngine:
    """Retrieves relevant chunks from the vector store based on a query."""

    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_dir: Union[str, Path],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the query engine.

        Args:
            index_path: Path to the FAISS index file.
            metadata_dir: Path to the directory containing metadata.json.
            model_name: The name of the embedding model to use.
        """
        self.index_path = Path(index_path)
        self.metadata_dir = Path(metadata_dir)

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
            
        if not (self.metadata_dir / "metadata.json").exists():
            raise FileNotFoundError(f"metadata.json not found in {self.metadata_dir}")

        self.embedder = Embedder(model_name=model_name)
        
        # Load the index
        self.vector_store = VectorStore()
        self.vector_store.load(self.index_path)
        
        # Load metadata
        _, self.metadata = load_embeddings(self.metadata_dir)
        
        if len(self.metadata) != self.vector_store.index.ntotal:
            raise ValueError(
                f"Mismatch between metadata count ({len(self.metadata)}) and "
                f"index size ({self.vector_store.index.ntotal})"
            )

    def query(self, question: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Query the vector store for the most relevant chunks.

        Args:
            question: The user's query string.
            k: The number of results to retrieve.

        Returns:
            A list of tuples, where each tuple contains:
                - The metadata dict for the retrieved chunk.
                - The distance/similarity score (lower is more similar for L2).
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        if not question.strip():
            return []
            
        # The Embedder takes a list of Chunk objects, but for a query we can
        # either mock a Chunk or just use the model directly.
        # It's cleaner to use the model directly here since a query isn't a document chunk.
        query_embedding = self.embedder.model.encode([question], show_progress_bar=False)
        
        distances, indices = self.vector_store.search(query_embedding, k=k)
        
        results = []
        if distances.size == 0 or indices.size == 0:
            return results
            
        # Extract the first (and only) query's results
        distances = distances[0]
        indices = indices[0]
        
        for dist, idx in zip(distances, indices):
            if idx == -1:
                # FAISS returns -1 if there are not enough elements in the index
                continue
                
            results.append((self.metadata[idx], float(dist)))
            
        return results
