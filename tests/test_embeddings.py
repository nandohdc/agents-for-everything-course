import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.chunker import Chunk
from src.embeddings import Embedder, load_embeddings, save_embeddings


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
    def test_embed_empty_chunks(self, mock_transformer_cls):
        """Test that empty inputs yield an empty array without calling the model."""
        mock_model = MagicMock()
        mock_transformer_cls.return_value = mock_model
        
        embedder = Embedder(model_name="dummy-model")
        embeddings = embedder.embed_chunks([])
        
        mock_model.encode.assert_not_called()
        self.assertEqual(embeddings.size, 0)
        self.assertEqual(embeddings.shape, (0,))

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
