import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from src.query_engine import QueryEngine


class TestQueryEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.index_path = self.base_path / "faiss.index"
        
        # Mock metadata
        self.metadata = [
            {"source": "doc1.txt", "text": "This is a test chunk."},
            {"source": "doc2.txt", "text": "Another chunk of text."}
        ]
        
        # Write metadata
        np.save(self.base_path / "embeddings.npy", np.random.rand(2, 384).astype(np.float32))
        with open(self.base_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
            
        # We need to mock VectorStore and Embedder so we don't actually use FAISS/SentenceTransformers
        # during the unit tests if we want them to be fast, but we can also just patch them.

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("src.query_engine.VectorStore")
    @patch("src.query_engine.Embedder")
    def test_query_engine_initialization(self, MockEmbedder, MockVectorStore):
        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.index.ntotal = 2
        
        # We should create a dummy index file so it passes the exists() check
        with open(self.index_path, "w") as f:
            f.write("dummy index")
            
        engine = QueryEngine(
            index_path=self.index_path,
            metadata_dir=self.base_path
        )
        

        self.assertEqual(len(engine.metadata), 2)
        mock_vs_instance.load.assert_called_once_with(self.index_path)

    @patch("src.query_engine.VectorStore")
    @patch("src.query_engine.Embedder")
    def test_query_engine_search(self, MockEmbedder, MockVectorStore):
        # Setup dummy index file
        with open(self.index_path, "w") as f:
            f.write("dummy index")

        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.index.ntotal = 2
        # Mock search to return one valid result and one -1
        mock_vs_instance.search.return_value = (
            np.array([[0.5, 0.9]]), 
            np.array([[1, -1]])
        )
        
        mock_embedder_instance = MockEmbedder.return_value
        mock_embedder_instance.model.encode.return_value = np.random.rand(1, 384)

        engine = QueryEngine(
            index_path=self.index_path,
            metadata_dir=self.base_path
        )
        
        results = engine.query("test query", k=2)
        
        # Should only return the valid index (1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["text"], "Another chunk of text.")
        self.assertEqual(results[0][1], 0.5)

    @patch("src.query_engine.VectorStore")
    @patch("src.query_engine.Embedder")
    def test_empty_query(self, MockEmbedder, MockVectorStore):
        with open(self.index_path, "w") as f:
            f.write("dummy index")

        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.index.ntotal = 2

        engine = QueryEngine(
            index_path=self.index_path,
            metadata_dir=self.base_path
        )
        
        results = engine.query("   ", k=2)
        self.assertEqual(len(results), 0)

    @patch("src.query_engine.VectorStore")
    @patch("src.query_engine.Embedder")
    def test_query_rejects_invalid_k(self, MockEmbedder, MockVectorStore):
        with open(self.index_path, "w") as f:
            f.write("dummy index")

        mock_vs_instance = MockVectorStore.return_value
        mock_vs_instance.index.ntotal = 2

        engine = QueryEngine(
            index_path=self.index_path,
            metadata_dir=self.base_path
        )

        with self.assertRaisesRegex(ValueError, "k must be > 0"):
            engine.query("test query", k=0)


if __name__ == "__main__":
    unittest.main()
