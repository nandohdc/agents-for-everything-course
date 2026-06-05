import io
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src import cli


def run_cli(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


class TestCliErrors(unittest.TestCase):
    def test_rejects_invalid_top_k_before_loading_models(self):
        code, stdout, stderr = run_cli(["Question?", "--top-k", "-1"])

        self.assertEqual(code, 2)
        self.assertIn("must be a positive integer", stderr)
        self.assertNotIn("Loading retrieval + generation models", stdout)
        self.assertNotIn("Traceback", stdout + stderr)

    def test_rejects_invalid_max_tokens_before_loading_models(self):
        code, stdout, stderr = run_cli(["Question?", "--max-tokens", "0"])

        self.assertEqual(code, 2)
        self.assertIn("must be a positive integer", stderr)
        self.assertNotIn("Loading retrieval + generation models", stdout)
        self.assertNotIn("Traceback", stdout + stderr)

    def test_missing_data_dir_returns_user_facing_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            code, stdout, stderr = run_cli(
                [
                    "How should I store coffee?",
                    "--index-file",
                    str(base / "index" / "faiss.index"),
                    "--metadata-dir",
                    str(base / "index"),
                    "--data-dir",
                    str(base / "missing"),
                    "--no-history",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("Corpus directory not found", stderr)
        self.assertNotIn("Traceback", stdout + stderr)

    def test_empty_data_dir_returns_user_facing_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "empty"
            data_dir.mkdir()
            code, stdout, stderr = run_cli(
                [
                    "How should I store coffee?",
                    "--index-file",
                    str(base / "index" / "faiss.index"),
                    "--metadata-dir",
                    str(base / "index"),
                    "--data-dir",
                    str(data_dir),
                    "--no-history",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("No documents found", stderr)
        self.assertNotIn("Traceback", stdout + stderr)

    def test_blank_data_dir_returns_user_facing_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "blank"
            data_dir.mkdir()
            (data_dir / "blank.txt").write_text("   \n\t\n", encoding="utf-8")
            code, stdout, stderr = run_cli(
                [
                    "How should I store coffee?",
                    "--index-file",
                    str(base / "index" / "faiss.index"),
                    "--metadata-dir",
                    str(base / "index"),
                    "--data-dir",
                    str(data_dir),
                    "--no-history",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("No chunks produced", stderr)
        self.assertNotIn("Traceback", stdout + stderr)

    def test_langchain_constructor_import_error_is_user_facing(self):
        fake_module = types.ModuleType("src.langchain_pipeline")

        class FakeLangChainRAG:
            def __init__(self, *args, **kwargs):
                raise ImportError("LangChain extras are not installed")

        fake_module.LangChainRAG = FakeLangChainRAG

        with patch.dict(sys.modules, {"src.langchain_pipeline": fake_module}):
            code, stdout, stderr = run_cli(
                [
                    "--engine",
                    "langchain",
                    "How should I store coffee beans?",
                    "--no-history",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("LangChain extras are not installed", stderr)
        self.assertNotIn("Traceback", stdout + stderr)


if __name__ == "__main__":
    unittest.main()
