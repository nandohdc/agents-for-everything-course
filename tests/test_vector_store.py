import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.dimension = 384
        self.vector_store = VectorStore(dimension=self.dimension)
        
    def test_initialization(self):
        self.assertEqual(self.vector_store.dimension, self.dimension)
        self.assertEqual(self.vector_store.index.ntotal, 0)
        
    def test_add_embeddings(self):
        embeddings = np.random.default_rng(42).random((5, self.dimension)).astype(np.float32)
        self.vector_store.add(embeddings)
        self.assertEqual(self.vector_store.index.ntotal, 5)
        
    def test_add_wrong_dimension(self):
        embeddings = np.random.default_rng(42).random((5, 128)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.vector_store.add(embeddings)
            
    def test_save_and_load(self):
        embeddings = np.random.default_rng(42).random((10, self.dimension)).astype(np.float32)
        self.vector_store.add(embeddings)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test.index"
            self.vector_store.save(index_path)
            
            self.assertTrue(index_path.exists())
            
            # Load into a new instance
            new_store = VectorStore(dimension=128) # Test that dimension updates correctly
            new_store.load(index_path)
            
            self.assertEqual(new_store.dimension, self.dimension)
            self.assertEqual(new_store.index.ntotal, 10)
            
    def test_search(self):
        embeddings = np.zeros((5, self.dimension), dtype=np.float32)
        # Make the 3rd embedding exactly match the query
        query_embedding = np.ones(self.dimension, dtype=np.float32)
        embeddings[2] = query_embedding
        
        self.vector_store.add(embeddings)
        
        distances, indices = self.vector_store.search(query_embedding, k=1)
        self.assertEqual(indices[0][0], 2)
        self.assertAlmostEqual(distances[0][0], 0.0)

    def test_search_empty(self):
        query_embedding = np.ones(self.dimension, dtype=np.float32)
        distances, indices = self.vector_store.search(query_embedding, k=1)
        self.assertEqual(distances.size, 0)
        self.assertEqual(indices.size, 0)
