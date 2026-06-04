# Repository Guidelines

## Project Structure & Module Organization

This repository contains a simple FAQ chatbot project built around a lightweight RAG pipeline.

- `src/`: Python package code for the chatbot, indexing, retrieval, and generation logic.
- `tests/`: Automated tests for source modules.
- `data/`: Local text corpora used as chatbot knowledge sources.
- `indexes/`: Generated FAISS indexes and related local artifacts.
- `scripts/`: Small CLI or utility scripts for setup, indexing, or demos.
- `final-project.md`: Project specification and expected architecture.
- `requirements.txt`: Python dependencies for embeddings, FAISS, generation, and numerical work.

Keep generated model caches and local virtual environments out of Git; `.gitignore` already covers common Python, Hugging Face, and Torch artifacts.

## Build, Test, and Development Commands

Create a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

There is no configured build system yet. Use targeted Python commands while developing, for example:

```bash
python -m pytest
python scripts/<script_name>.py
```

If a command requires downloaded models or generated indexes, document the expected input files and outputs in the script or README.

## Coding Style & Naming Conventions

Write idiomatic Python with 4-space indentation and clear module boundaries. Use `snake_case` for files, functions, and variables; use `PascalCase` for classes. Prefer small, testable functions for embedding, indexing, retrieval, and generation steps.

No formatter or linter is configured yet. Before adding new tooling, keep changes minimal and consistent with the existing project scope.

## Testing Guidelines

Place tests under `tests/` and name files `test_<module>.py`. Use small fixtures and avoid requiring large model downloads in unit tests. For model-dependent behavior, prefer smoke tests or documented manual checks unless lightweight mocks are practical.

Run tests with:

```bash
python -m pytest
```

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit messages such as `Scaffold project structure` and `Define dependencies and project configuration`. Keep commits focused on one issue or behavior change.

Pull requests should include a concise summary, linked issue such as `Closes #2`, validation commands run, and any limitations. Include screenshots only for user-facing UI changes.

## Agent-Specific Instructions

Preserve the simple RAG architecture described in `final-project.md`: local corpus, MiniLM embeddings, FAISS retrieval, and a lightweight generation interface. Avoid paid APIs or unrelated framework additions unless an issue explicitly requires them.
