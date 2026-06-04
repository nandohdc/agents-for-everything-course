"""Tests for src.document_loader using standard-library unittest."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.document_loader import Document, load_documents


class TestLoadDocumentsWithFixture(unittest.TestCase):
    """Test load_documents against a temporary corpus fixture."""

    def setUp(self) -> None:
        """Create a temp directory tree with .txt, nested .txt, and non-.txt files."""
        self.tmpdir = tempfile.mkdtemp()

        # Top-level .txt file
        self._write(os.path.join(self.tmpdir, "alpha.txt"), "Alpha content")

        # Nested .txt file
        nested_dir = os.path.join(self.tmpdir, "sub")
        os.makedirs(nested_dir)
        self._write(os.path.join(nested_dir, "beta.txt"), "Beta content")

        # Non-.txt file (should be ignored)
        self._write(os.path.join(self.tmpdir, "ignore.md"), "Markdown file")

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_correct_document_count(self) -> None:
        docs = load_documents(self.tmpdir)
        self.assertEqual(len(docs), 2)

    def test_non_txt_files_excluded(self) -> None:
        docs = load_documents(self.tmpdir)
        filenames = [d.metadata["filename"] for d in docs]
        self.assertNotIn("ignore.md", filenames)

    def test_deterministic_ordering(self) -> None:
        docs = load_documents(self.tmpdir)
        paths = [d.metadata["path"] for d in docs]
        self.assertEqual(paths, sorted(paths))

    def test_text_contents(self) -> None:
        docs = load_documents(self.tmpdir)
        texts = {d.metadata["filename"]: d.text for d in docs}
        self.assertEqual(texts["alpha.txt"], "Alpha content")
        self.assertEqual(texts["beta.txt"], "Beta content")

    def test_filename_metadata(self) -> None:
        docs = load_documents(self.tmpdir)
        filenames = {d.metadata["filename"] for d in docs}
        self.assertEqual(filenames, {"alpha.txt", "beta.txt"})

    def test_relative_path_metadata(self) -> None:
        docs = load_documents(self.tmpdir)
        dir_name = Path(self.tmpdir).name
        for doc in docs:
            self.assertTrue(
                doc.metadata["path"].startswith(dir_name + "/"),
                f"Expected path to start with '{dir_name}/', got '{doc.metadata['path']}'",
            )


class TestLoadDocumentsWithDefaultCorpus(unittest.TestCase):
    """Validate loading against the real data/ corpus shipped in the repo."""

    def test_loads_all_txt_files(self) -> None:
        """The default data/ directory should contain every .txt file present."""
        docs = load_documents("data")
        data_dir = Path("data")
        expected_count = len(list(data_dir.rglob("*.txt")))
        self.assertGreater(expected_count, 0, "data/ should contain at least one .txt file")
        self.assertEqual(len(docs), expected_count)

    def test_metadata_keys_present(self) -> None:
        docs = load_documents("data")
        for doc in docs:
            self.assertIn("filename", doc.metadata)
            self.assertIn("path", doc.metadata)

    def test_paths_are_relative(self) -> None:
        docs = load_documents("data")
        for doc in docs:
            self.assertTrue(
                doc.metadata["path"].startswith("data/"),
                f"Expected relative path starting with 'data/', got '{doc.metadata['path']}'",
            )
            self.assertFalse(
                os.path.isabs(doc.metadata["path"]),
                "Path metadata should not be absolute",
            )


class TestLoadDocumentsErrors(unittest.TestCase):
    """Validate error handling for invalid corpus paths."""

    def test_missing_directory_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_documents("/nonexistent/path/to/corpus")

    def test_file_instead_of_directory_raises_not_a_directory(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with self.assertRaises(NotADirectoryError):
                load_documents(tmp.name)


class TestDocumentDataclass(unittest.TestCase):
    """Validate the Document dataclass behaviour."""

    def test_frozen(self) -> None:
        doc = Document(text="hello", metadata={"filename": "a.txt", "path": "data/a.txt"})
        with self.assertRaises(AttributeError):
            doc.text = "modified"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
