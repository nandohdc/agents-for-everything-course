"""Embedding generation and persistence for the RAG pipeline.

Provides tools to generate embeddings from text chunks using SentenceTransformers,
and utilities to save/load the resulting embeddings and their metadata to disk.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Wrapper around SentenceTransformer for generating chunk embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the embedder by loading the model.

        Args:
            model_name: The Hugging Face model identifier or path.
        """
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate dense vector embeddings for a list of chunks.

        Args:
            chunks: The chunks to embed.

        Returns:
            A numpy array of embeddings with shape (num_chunks, embedding_dim).
            If the input list is empty, returns an empty array.
        """
        if not chunks:
            return np.array([])
            
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings


def chunk_to_metadata(chunk: Chunk) -> Dict[str, Any]:
    """Build a persist-ready metadata dict for a chunk.

    The chunker stores positional metadata (``source_filename``, ``source_path``,
    ``chunk_index``, ``chunk_size``) on ``chunk.metadata`` and keeps the chunk text
    on ``chunk.text``. Retrieval consumers (``scripts/query.py``,
    ``src/prompt_builder.py``) need the chunk ``text`` and a display ``source`` to
    surface results and build prompts, so enrich the metadata with both here before
    persisting it alongside the embeddings.

    Args:
        chunk: The source chunk.

    Returns:
        A new dict containing the chunk's original metadata plus ``text`` (the chunk
        body) and ``source`` (its source path, falling back to filename).
    """
    metadata = dict(chunk.metadata)
    metadata["text"] = chunk.text
    metadata.setdefault(
        "source",
        chunk.metadata.get("source_path")
        or chunk.metadata.get("source_filename", "Unknown"),
    )
    return metadata


def save_embeddings(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: Union[str, Path]
) -> None:
    """Save embeddings and associated metadata to a directory.

    Creates the directory if it does not exist. The embeddings are saved as
    `embeddings.npy` and the metadata as `metadata.json`.

    Args:
        embeddings: The numpy array of embeddings.
        metadata: A list of metadata dicts corresponding to the embedded chunks.
        output_dir: Path to the directory where files will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_embeddings(
    input_dir: Union[str, Path]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load embeddings and associated metadata from a directory.

    Args:
        input_dir: Path to the directory containing `embeddings.npy` and `metadata.json`.

    Returns:
        A tuple containing:
            - A numpy array of embeddings.
            - A list of metadata dicts.
    """
    input_dir = Path(input_dir)
    
    embeddings = np.load(input_dir / "embeddings.npy")
    with open(input_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return embeddings, metadata
