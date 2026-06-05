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
    def test_parses_hf_token(self):
        args = cli.build_arg_parser().parse_args(
            ["How should I store coffee?", "--hf-token", "hf_test"]
        )

        self.assertEqual(args.hf_token, "hf_test")

    @patch("src.cli._dispatch", return_value=0)
    @patch("src.cli.Generator")
    @patch("src.cli.QueryEngine")
    @patch(
        "src.cli.build_index",
        return_value={"num_chunks": 1, "num_documents": 1, "dimension": 384},
    )
    @patch("src.cli.index_exists", return_value=False)
    def test_hf_token_is_passed_to_baseline_components(
        self,
        mock_index_exists,
        mock_build_index,
        mock_query_engine,
        mock_generator,
        mock_dispatch,
    ):
        code, stdout, stderr = run_cli(
            [
                "How should I store coffee?",
                "--hf-token",
                "hf_test",
                "--index-file",
                "tmp/faiss.index",
                "--metadata-dir",
                "tmp",
                "--no-history",
            ]
        )

        self.assertEqual(code, 0)
        mock_index_exists.assert_called_once_with("tmp/faiss.index", "tmp")
        mock_build_index.assert_called_once_with(
            data_dir="data",
            output_dir="tmp",
            index_file="tmp/faiss.index",
            hf_token="hf_test",
        )
        mock_query_engine.assert_called_once_with(
            index_path="tmp/faiss.index",
            metadata_dir="tmp",
            model_name=cli.EMBEDDING_MODEL,
            hf_token="hf_test",
        )
        mock_generator.assert_called_once_with(
            model_name=cli.DEFAULT_MODEL, hf_token="hf_test"
        )
        mock_dispatch.assert_called_once()
        self.assertNotIn("hf_test", stdout + stderr)

    @patch("src.cli._dispatch", return_value=0)
    def test_hf_token_is_passed_to_langchain_backend(self, mock_dispatch):
        fake_module = types.ModuleType("src.langchain_pipeline")
        constructed_kwargs = {}

        class FakeLangChainRAG:
            def __init__(self, *args, **kwargs):
                constructed_kwargs.update(kwargs)

            def answer(self, question):
                return "answer", []

        fake_module.LangChainRAG = FakeLangChainRAG

        with patch.dict(sys.modules, {"src.langchain_pipeline": fake_module}):
            code, stdout, stderr = run_cli(
                [
                    "--engine",
                    "langchain",
                    "How should I store coffee?",
                    "--hf-token",
                    "hf_test",
                    "--no-history",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(constructed_kwargs["hf_token"], "hf_test")
        mock_dispatch.assert_called_once()
        self.assertNotIn("hf_test", stdout + stderr)

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
