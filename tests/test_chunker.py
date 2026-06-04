"""Tests for src.chunker using standard-library unittest."""

from __future__ import annotations

import unittest

from src.chunker import (
    Chunk,
    _normalize_whitespace,
    chunk_document,
    chunk_documents,
)
from src.document_loader import Document, load_documents


def _make_doc(text: str, filename: str = "sample.txt") -> Document:
    return Document(
        text=text,
        metadata={"filename": filename, "path": f"data/{filename}"},
    )


class TestChunkDocumentBasics(unittest.TestCase):
    """Core chunking behaviour against a deterministic, whitespace-free body."""

    def setUp(self) -> None:
        # 1000 chars, no internal whitespace, so slice boundaries are exact.
        self.text = "abcdefghij" * 100
        self.doc = _make_doc(self.text)
        self.chunk_size = 100
        self.chunk_overlap = 20
        self.chunks = chunk_document(self.doc, self.chunk_size, self.chunk_overlap)

    def test_expected_chunk_count(self) -> None:
        # step = 80; starts at 0,80,...,960 -> 13 chunks.
        self.assertEqual(len(self.chunks), 13)

    def test_chunk_sizes_within_limit(self) -> None:
        for chunk in self.chunks:
            self.assertLessEqual(len(chunk.text), self.chunk_size)
            self.assertEqual(chunk.metadata["chunk_size"], len(chunk.text))

    def test_full_chunks_are_exact_size(self) -> None:
        # Every chunk except the last should be exactly chunk_size long.
        for chunk in self.chunks[:-1]:
            self.assertEqual(len(chunk.text), self.chunk_size)

    def test_consecutive_chunks_overlap(self) -> None:
        for first, second in zip(self.chunks, self.chunks[1:]):
            if len(first.text) == self.chunk_size:
                self.assertEqual(
                    first.text[-self.chunk_overlap :],
                    second.text[: self.chunk_overlap],
                )

    def test_chunks_reconstruct_source(self) -> None:
        # Dropping the overlap from each subsequent chunk rebuilds the text.
        step = self.chunk_size - self.chunk_overlap
        rebuilt = self.chunks[0].text + "".join(
            c.text[self.chunk_overlap :] for c in self.chunks[1:]
        )
        # Sanity: step-based reconstruction matches the normalized source.
        self.assertEqual(rebuilt, self.text)
        self.assertEqual(step, 80)

    def test_chunk_index_is_sequential(self) -> None:
        indices = [c.metadata["chunk_index"] for c in self.chunks]
        self.assertEqual(indices, list(range(len(self.chunks))))

    def test_source_metadata_preserved(self) -> None:
        for chunk in self.chunks:
            self.assertEqual(chunk.metadata["source_filename"], "sample.txt")
            self.assertEqual(chunk.metadata["source_path"], "data/sample.txt")


class TestWhitespaceNormalization(unittest.TestCase):
    def test_normalize_collapses_and_strips(self) -> None:
        self.assertEqual(
            _normalize_whitespace("  hello\t\tworld\n\nfoo  "),
            "hello world foo",
        )

    def test_chunk_text_is_normalized(self) -> None:
        doc = _make_doc("alpha   beta\n\n\tgamma")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "alpha beta gamma")


class TestEdgeCases(unittest.TestCase):
    def test_empty_document_yields_no_chunks(self) -> None:
        self.assertEqual(chunk_document(_make_doc("")), [])

    def test_whitespace_only_document_yields_no_chunks(self) -> None:
        self.assertEqual(chunk_document(_make_doc("   \n\t  ")), [])

    def test_document_shorter_than_chunk_size(self) -> None:
        doc = _make_doc("short text")
        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short text")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_single_character_document(self) -> None:
        chunks = chunk_document(_make_doc("x"), chunk_size=10, chunk_overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "x")

    def test_text_exactly_chunk_size(self) -> None:
        chunks = chunk_document(_make_doc("a" * 100), chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].text), 100)


class TestInvalidParameters(unittest.TestCase):
    def test_zero_chunk_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(_make_doc("text"), chunk_size=0)

    def test_negative_chunk_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(_make_doc("text"), chunk_size=-5)

    def test_negative_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(_make_doc("text"), chunk_size=100, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(_make_doc("text"), chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(_make_doc("text"), chunk_size=100, chunk_overlap=150)


class TestChunkDocuments(unittest.TestCase):
    def test_wrapper_concatenates_in_order(self) -> None:
        docs = [
            _make_doc("a" * 250, filename="a.txt"),
            _make_doc("b" * 250, filename="b.txt"),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        per_doc = [chunk_document(d, 100, 20) for d in docs]
        self.assertEqual(len(chunks), sum(len(c) for c in per_doc))
        # Index restarts per source document.
        a_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["source_filename"] == "a.txt"]
        self.assertEqual(a_indices, list(range(len(per_doc[0]))))

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(chunk_documents([]), [])


class TestChunkDataclass(unittest.TestCase):
    def test_frozen(self) -> None:
        chunk = Chunk(text="hi", metadata={})
        with self.assertRaises(AttributeError):
            chunk.text = "modified"  # type: ignore[misc]


class TestRealCorpusIntegration(unittest.TestCase):
    """Chunk the real data/ corpus shipped in the repo."""

    def test_chunks_real_corpus(self) -> None:
        docs = load_documents("data")
        self.assertGreater(len(docs), 0, "data/ should contain at least one .txt file")

        chunks = chunk_documents(docs)
        self.assertGreater(len(chunks), 0)

        sources = {d.metadata["path"] for d in docs}
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 500)
            self.assertIn(chunk.metadata["source_path"], sources)
            self.assertIn("source_filename", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)
            # Normalization guarantees no double spaces or stray newlines.
            self.assertNotIn("\n", chunk.text)
            self.assertNotIn("  ", chunk.text)


if __name__ == "__main__":
    unittest.main()
