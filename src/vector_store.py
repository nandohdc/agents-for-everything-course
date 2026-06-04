"""Vector store management using FAISS."""

import faiss
import numpy as np
from pathlib import Path
from typing import Union, Tuple


class VectorStore:
    def __init__(self, dimension: int = 384):
        """Initialize the FAISS index.
        
        Args:
            dimension: The dimension of the embeddings (default is 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index.
        
        Args:
            embeddings: A numpy array of embeddings with shape (num_chunks, dimension).
        """
        if embeddings.size == 0:
            return
            
        # FAISS requires float32
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings_np.shape) == 1:
            embeddings_np = np.expand_dims(embeddings_np, axis=0)
            
        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(f"Expected embeddings of dimension {self.dimension}, got {embeddings_np.shape[1]}")
            
        self.index.add(embeddings_np)
        
    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the index to disk.
        
        Args:
            filepath: The path to save the index to.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(filepath))
        
    def load(self, filepath: Union[str, Path]) -> None:
        """Load an index from disk.
        
        Args:
            filepath: The path to load the index from.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found at {filepath}")
            
        self.index = faiss.read_index(str(filepath))
        self.dimension = self.index.d
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for the top k most similar embeddings.
        
        Args:
            query_embedding: A numpy array representing the query embedding.
            k: The number of results to return.
            
        Returns:
            A tuple containing:
                - distances: A numpy array of distances to the returned embeddings.
                - indices: A numpy array of the indices of the returned embeddings.
        """
        if self.index.ntotal == 0:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
            
        query_np = np.asarray(query_embedding, dtype=np.float32)
        if len(query_np.shape) == 1:
            query_np = np.expand_dims(query_np, axis=0)
            
        distances, indices = self.index.search(query_np, k)
        return distances, indices
