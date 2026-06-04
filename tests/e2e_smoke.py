"""End-to-end smoke test for the coffee FAQ RAG pipeline (Issue #12).

This is a standalone script: its filename does not match ``test*.py``, so it is
intentionally **excluded** from ``python -m unittest discover`` (keeping the
unit suite free of model downloads, per the course guideline). It exercises the
full pipeline against the real ``data/`` corpus and downloads the MiniLM +
flan-t5 weights on first run.

Run from the repo root::

    python tests/e2e_smoke.py

Exits 0 on success (indexing + retrieval + a non-empty answer) and 1 on failure.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Ensure the project root is on sys.path so ``src`` can be imported when running
# this script directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.cli import answer_question, build_index, result_sources  # noqa: E402
from src.generator import Generator  # noqa: E402
from src.query_engine import QueryEngine  # noqa: E402

SAMPLE_QUESTION = "How should I store coffee beans to keep them fresh?"


def run() -> int:
    data_dir = Path(_project_root) / "data"

    with TemporaryDirectory() as tmp:
        index_dir = Path(tmp)
        index_file = index_dir / "faiss.index"

        # 1. Indexing
        print(f"[1/3] Building index from {data_dir} ...")
        summary = build_index(
            data_dir=data_dir, output_dir=index_dir, index_file=index_file
        )
        print(
            f"      indexed {summary['num_chunks']} chunks from "
            f"{summary['num_documents']} docs (dim={summary['dimension']})."
        )
        assert summary["num_chunks"] > 0, "indexing produced no chunks"

        # 2. Retrieval
        print("[2/3] Loading query engine + generator ...")
        engine = QueryEngine(index_path=index_file, metadata_dir=index_dir)
        generator = Generator()

        # 3. Generation
        print(f"[3/3] Answering: {SAMPLE_QUESTION!r}")
        answer, results = answer_question(
            SAMPLE_QUESTION, engine, generator, k=3
        )

        assert results, "retrieval returned no chunks"
        assert answer and answer.strip(), "generation returned an empty answer"

        print("\n--- Retrieved sources ---")
        for metadata, score in results:
            print(f"  ({score:.4f}) {metadata.get('source', 'Unknown')}")
        print(f"\n--- Answer ---\n{answer}\n")
        print(f"Sources: {', '.join(result_sources(results))}")

    print("\nE2E smoke test PASSED.")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 - smoke test reports any failure
        print(f"E2E smoke test FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
