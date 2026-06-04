# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Final project for the "Agents for Everything" course: a terminal FAQ chatbot over a local
coffee-domain corpus, built as a **lightweight RAG pipeline**. The full spec lives in
`final-project.md` — read it before adding pipeline stages.

The intended architecture (mostly not yet built) is a 4-stage pipeline:
1. **Load** local `.txt` corpus from `data/` → `src/document_loader.py` (done).
2. **Index** — embed chunks with `sentence-transformers/all-MiniLM-L6-v2`, store in a FAISS index under `indexes/`.
3. **Retrieve** — FAISS similarity search for the most relevant excerpts to a question.
4. **Generate** — answer from retrieved context with a free/local model (e.g. `flan-t5-base`).

Today only stage 1 exists. Stages 2–4, plus the CLI/notebook interface, are still to come.

## Hard constraints (from the course spec)

- **No paid APIs.** Use only local models or free Hugging Face endpoints.
- Keep the pipeline simple: FAISS + MiniLM embeddings + a lightweight generator. Don't pull in
  heavyweight frameworks unless an issue explicitly calls for it (LangChain is an *optional* bonus).
- Dependency set is validated against **Python 3.14**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Testing

Tests use the **standard-library `unittest`** module. Note that **pytest is not in
`requirements.txt`** and is not installed by default, so `unittest` is the reliable runner:

```bash
python -m unittest discover -s tests                          # all tests
python -m unittest tests.test_document_loader                 # one file
python -m unittest tests.test_document_loader.TestLoadDocumentsErrors  # one class
python -m unittest tests.test_document_loader.TestDocumentDataclass.test_frozen  # one test
```

`python -m pytest` also works *if* you `pip install pytest` first (the suite is pytest-compatible),
but don't assume it's available.

Tests must run with the **repo root as CWD** — `TestLoadDocumentsWithDefaultCorpus` loads the real
`data/` corpus by relative path, and `src` is imported as a top-level package.

Keep unit tests free of large model downloads (course guideline). For embedding/generation
behavior, prefer smoke tests or mocks over loading real MiniLM/flan-t5 weights.

## Running the document loader

```bash
python scripts/load_documents.py                  # loads data/, prints a summary
python scripts/load_documents.py --data-dir path  # custom corpus dir
```

Scripts in `scripts/` are run directly, so they prepend the project root to `sys.path` themselves
(see the top of `load_documents.py`) to import `src`. Match that pattern for new scripts rather than
relying on installed packaging.

## Conventions

- `load_documents()` returns frozen `Document` dataclasses (immutable `text` + `metadata`), sorted
  by relative POSIX path for **deterministic ordering** — downstream indexing depends on this; don't
  break it. Metadata always carries `filename` and `path` (path is relative to the corpus parent,
  e.g. `data/foo.txt`).
- `indexes/` holds generated FAISS artifacts; treat as build output, keep out of Git logic.
- Commits: short imperative subjects (e.g. `Implement document loading`). PRs link their issue
  (`Closes #N`) and list the validation commands run.
