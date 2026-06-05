"""Terminal CLI for the coffee FAQ RAG chatbot.

Wires the full pipeline into a single flow: **indexing** (build or load a FAISS
index from ``data/``) -> **retrieval** (:class:`~src.query_engine.QueryEngine`)
-> **generation** (:class:`~src.generator.Generator`) -> a printed answer with
its sources.

Run it as a module from the repo root::

    python -m src.cli "How should I store coffee beans?"   # single-shot
    python -m src.cli                                       # interactive REPL

If the FAISS index is missing it is built automatically from ``--data-dir``.
An optional LangChain backend is available via ``--engine langchain`` (see
``src/langchain_pipeline.py``). Each interaction is appended to
``history/qa_history.jsonl`` unless ``--no-history`` is given.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

# Ensure the project root is on sys.path so ``src`` can be imported when this
# module is executed directly. ``python -m src.cli`` already handles this; this
# keeps parity with the scripts/ entry points.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.chunker import chunk_documents  # noqa: E402
from src.document_loader import load_documents  # noqa: E402
from src.embeddings import Embedder, chunk_to_metadata, save_embeddings  # noqa: E402
from src.generator import Generator  # noqa: E402
from src.grounding import REFUSAL_ANSWER, context_supports_question  # noqa: E402
from src.history import append_interaction  # noqa: E402
from src.prompt_builder import build_prompt  # noqa: E402
from src.query_engine import QueryEngine  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "indexes"
DEFAULT_INDEX_FILE = "indexes/faiss.index"
DEFAULT_MODEL = "google/flan-t5-base"
DEFAULT_HISTORY_FILE = "history/qa_history.jsonl"
DEFAULT_TOP_K = 3
DEFAULT_MAX_TOKENS = 128
USER_FACING_EXCEPTIONS = (
    FileNotFoundError,
    NotADirectoryError,
    ValueError,
    ImportError,
)

# Type alias: a backend answers a question with (answer_text, source_labels).
AnswerFn = Callable[[str], Tuple[str, List[str]]]


def build_index(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    output_dir: Union[str, Path] = DEFAULT_INDEX_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    index_file: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Build and persist embeddings + a FAISS index from a text corpus.

    Loads every document under ``data_dir``, chunks it, embeds the chunks with
    ``all-MiniLM-L6-v2``, writes ``embeddings.npy``/``metadata.json`` to
    ``output_dir``, then builds and saves the FAISS index to ``index_file``
    (defaults to ``output_dir/faiss.index``).

    Args:
        data_dir: Directory of ``.txt`` documents to index.
        output_dir: Directory to write embeddings + metadata into.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        index_file: Path for the FAISS index file.

    Returns:
        A summary dict with ``num_documents``, ``num_chunks``, ``dimension``
        and ``index_file``.

    Raises:
        ValueError: If no documents or no chunks are produced.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    index_file = Path(index_file) if index_file else output_dir / "faiss.index"

    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No documents found in '{data_dir}'.")

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks produced from documents in '{data_dir}'.")

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    metadata = [chunk_to_metadata(chunk) for chunk in chunks]
    save_embeddings(embeddings, metadata, output_dir)

    dimension = int(embeddings.shape[1])
    store = VectorStore(dimension=dimension)
    store.add(embeddings)
    store.save(index_file)

    return {
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "dimension": dimension,
        "index_file": str(index_file),
    }


def index_exists(
    index_file: Union[str, Path], metadata_dir: Union[str, Path]
) -> bool:
    """Return ``True`` if both the FAISS index and its metadata are present."""
    return Path(index_file).exists() and (Path(metadata_dir) / "metadata.json").exists()


def result_sources(results: List[Tuple[Dict[str, Any], float]]) -> List[str]:
    """Ordered, de-duplicated list of source labels from retrieval results."""
    sources: List[str] = []
    for metadata, _ in results:
        source = metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)
    return sources


def positive_int(value: str) -> int:
    """Argparse type for positive integer CLI options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


def answer_question(
    question: str,
    engine: QueryEngine,
    generator: Generator,
    k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
    """Answer a question with the baseline pipeline.

    Retrieves the top-k chunks, builds a context-grounded prompt and generates
    an answer (greedy decoding for deterministic, context-faithful output).

    Returns:
        ``(answer, results)`` where ``results`` is the list of
        ``(metadata, distance)`` tuples from the query engine.
    """
    results = engine.query(question, k=k)
    if not context_supports_question(
        question, (metadata.get("text", "") for metadata, _ in results)
    ):
        return REFUSAL_ANSWER, results

    prompt = build_prompt(question, results)
    answer = generator.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
    return answer, results


def _print_answer(answer: str, sources: List[str]) -> None:
    print(f"\nAnswer: {answer}\n")
    if sources:
        print("Sources:")
        for source in sources:
            print(f"  - {source}")
    print()


def _maybe_save_history(
    history_file: Union[str, Path, None],
    question: str,
    answer: str,
    sources: List[str],
) -> None:
    """Append the interaction to history, never letting an error break the flow."""
    if not history_file:
        return
    try:
        append_interaction(history_file, question, answer, sources)
    except Exception as exc:  # pragma: no cover - defensive; history is optional
        print(f"(warning: could not write history to {history_file}: {exc})")


def _run_repl(answer_fn: AnswerFn, history_file: Union[str, Path, None]) -> None:
    """Interactive loop until the user types ``exit``/``quit`` or sends EOF."""
    print("Coffee FAQ chatbot. Ask a question, or type 'exit'/'quit' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer, sources = answer_fn(question)
        _print_answer(answer, sources)
        _maybe_save_history(history_file, question, answer, sources)
    print("Goodbye!")


def _dispatch(
    question: Union[str, None],
    answer_fn: AnswerFn,
    history_file: Union[str, Path, None],
) -> int:
    """Single-shot if a question was given, otherwise an interactive REPL."""
    if question:
        answer, sources = answer_fn(question)
        _print_answer(answer, sources)
        _maybe_save_history(history_file, question, answer, sources)
    else:
        _run_repl(answer_fn, history_file)
    return 0


def _run_baseline(args: argparse.Namespace, history_file: Union[str, Path, None]) -> int:
    # Ensure the index exists, building it from the corpus if needed.
    try:
        if args.rebuild or not index_exists(args.index_file, args.metadata_dir):
            if args.rebuild:
                print(f"Rebuilding index from '{args.data_dir}' ...", flush=True)
            else:
                print(
                    f"Index not found at '{args.index_file}'. "
                    f"Building it from '{args.data_dir}' ...",
                    flush=True,
                )
            summary = build_index(
                data_dir=args.data_dir,
                output_dir=args.metadata_dir,
                index_file=args.index_file,
            )
            print(
                f"Indexed {summary['num_chunks']} chunks from "
                f"{summary['num_documents']} documents (dim={summary['dimension']}).\n"
            )

        print("Loading retrieval + generation models (first run may download weights) ...")
        engine = QueryEngine(
            index_path=args.index_file,
            metadata_dir=args.metadata_dir,
            model_name=EMBEDDING_MODEL,
        )
        generator = Generator(model_name=args.model)

        def answer_fn(question: str) -> Tuple[str, List[str]]:
            answer, results = answer_question(
                question, engine, generator, k=args.top_k, max_tokens=args.max_tokens
            )
            return answer, result_sources(results)

        return _dispatch(args.question, answer_fn, history_file)
    except USER_FACING_EXCEPTIONS as exc:
        _print_error(str(exc))
        return 1


def _run_langchain(args: argparse.Namespace, history_file: Union[str, Path, None]) -> int:
    try:
        from src.langchain_pipeline import LangChainRAG
    except ImportError as exc:
        print(
            "LangChain mode requires the optional extras. Install them with:\n"
            "    pip install -r requirements-langchain.txt\n"
            f"(import error: {exc})"
        )
        return 1

    try:
        print("Building LangChain pipeline (first run may download weights) ...", flush=True)
        rag = LangChainRAG(
            data_dir=args.data_dir,
            k=args.top_k,
            max_tokens=args.max_tokens,
            model_name=args.model,
        )
        return _dispatch(args.question, rag.answer, history_file)
    except USER_FACING_EXCEPTIONS as exc:
        _print_error(str(exc))
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions about the local coffee corpus (RAG chatbot)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer. If omitted, an interactive session starts.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Corpus directory used to build the index (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--index-file",
        default=DEFAULT_INDEX_FILE,
        help=f"Path to the FAISS index (default: {DEFAULT_INDEX_FILE}).",
    )
    parser.add_argument(
        "--metadata-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing metadata.json (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max new tokens to generate (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Generator model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--engine",
        choices=["baseline", "langchain"],
        default="baseline",
        help="Pipeline backend (default: baseline).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from --data-dir before answering.",
    )
    parser.add_argument(
        "--history-file",
        default=DEFAULT_HISTORY_FILE,
        help=f"JSONL file to append Q&A history to (default: {DEFAULT_HISTORY_FILE}).",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable Q&A history storage.",
    )
    return parser


def main(argv: Union[List[str], None] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    history_file = None if args.no_history else args.history_file

    if args.engine == "langchain":
        return _run_langchain(args, history_file)
    return _run_baseline(args, history_file)


if __name__ == "__main__":
    raise SystemExit(main())
