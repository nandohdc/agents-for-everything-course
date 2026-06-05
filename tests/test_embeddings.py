import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.chunker import Chunk
from src.embeddings import (
    Embedder,
    chunk_to_metadata,
    load_embeddings,
    save_embeddings,
)


class TestEmbeddings(unittest.TestCase):
    @patch("src.embeddings.SentenceTransformer")
    def test_embed_chunks(self, mock_transformer_cls):
        """Test that chunks are embedded correctly using the mock model."""
        mock_model = MagicMock()
        # Provide deterministic mock embeddings for our 2 input chunks
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_transformer_cls.return_value = mock_model

        embedder = Embedder(model_name="dummy-model")

        chunks = [
            Chunk(text="Hello", metadata={"id": 1}),
            Chunk(text="World", metadata={"id": 2})
        ]

        embeddings = embedder.embed_chunks(chunks)

        # Verify that encode was called with just the text content
        mock_model.encode.assert_called_once_with(
            ["Hello", "World"], 
            show_progress_bar=False
        )
        self.assertEqual(embeddings.shape, (2, 3))
        self.assertTrue(np.allclose(embeddings[0], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(embeddings[1], [0.4, 0.5, 0.6]))

    @patch("src.embeddings.SentenceTransformer")
    def test_initialization_passes_hf_token(self, mock_transformer_cls):
        """Test that an explicit Hugging Face token is passed to the model."""
        Embedder(model_name="private-model", hf_token="hf_test")

        mock_transformer_cls.assert_called_once_with(
            "private-model", token="hf_test"
        )

    @patch("src.embeddings.SentenceTransformer")
    def test_embed_empty_chunks(self, mock_transformer_cls):
        """Test that empty inputs yield an empty array without calling the model."""
        mock_model = MagicMock()
        mock_transformer_cls.return_value = mock_model
        
        embedder = Embedder(model_name="dummy-model")
        embeddings = embedder.embed_chunks([])
        
        mock_model.encode.assert_not_called()
        self.assertEqual(embeddings.size, 0)
        self.assertEqual(embeddings.shape, (0,))

    def test_chunk_to_metadata_enriches_with_text_and_source(self):
        """Persisted metadata must carry the chunk text and a display source.

        Regression for the retrieval gap: the chunker keeps text on ``Chunk.text``
        and uses ``source_path``/``source_filename`` keys, but downstream consumers
        (query script, prompt builder) read ``text`` and ``source``.
        """
        chunk = Chunk(
            text="Store beans in an airtight container.",
            metadata={
                "source_filename": "coffee_storage_and_freshness.txt",
                "source_path": "data/coffee_storage_and_freshness.txt",
                "chunk_index": 2,
                "chunk_size": 37,
            },
        )

        metadata = chunk_to_metadata(chunk)

        # Chunk body is now retrievable via the "text" key.
        self.assertEqual(metadata["text"], "Store beans in an airtight container.")
        # A non-empty display source is present (prefers the relative path).
        self.assertEqual(metadata["source"], "data/coffee_storage_and_freshness.txt")
        # Original chunk metadata keys are preserved.
        self.assertEqual(metadata["source_filename"], "coffee_storage_and_freshness.txt")
        self.assertEqual(metadata["chunk_index"], 2)
        self.assertEqual(metadata["chunk_size"], 37)
        # The helper returns a new dict; it does not mutate the chunk's metadata.
        self.assertNotIn("text", chunk.metadata)

    def test_chunk_to_metadata_source_falls_back_to_filename(self):
        """When no source_path is available, source falls back to the filename."""
        chunk = Chunk(text="hello", metadata={"source_filename": "a.txt"})

        metadata = chunk_to_metadata(chunk)

        self.assertEqual(metadata["source"], "a.txt")
        self.assertEqual(metadata["text"], "hello")

    def test_save_and_load_embeddings(self):
        """Test persistence of embeddings and metadata to disk."""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        metadata = [{"id": 1}, {"id": 2}]

        with tempfile.TemporaryDirectory() as temp_dir:
            save_embeddings(embeddings, metadata, temp_dir)

            # Check files were created
            temp_path = Path(temp_dir)
            self.assertTrue((temp_path / "embeddings.npy").exists())
            self.assertTrue((temp_path / "metadata.json").exists())

            # Load them back
            loaded_embeddings, loaded_metadata = load_embeddings(temp_dir)

            self.assertTrue(np.allclose(embeddings, loaded_embeddings))
            self.assertEqual(metadata, loaded_metadata)


if __name__ == "__main__":
    unittest.main()
