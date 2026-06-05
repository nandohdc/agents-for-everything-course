# Copilot instructions

## Build, test, lint
- Install baseline dependencies with `python -m pip install -r requirements.txt`.
- Run the fast unit suite with `python -m unittest discover -s tests`.
- Run the real end-to-end smoke test with `python tests/e2e_smoke.py`.
- Optional LangChain mode requires `python -m pip install -r requirements-langchain.txt`.
- No formatter or linter is configured; keep changes idiomatic and minimal.

## Project overview
- This repository contains a terminal coffee FAQ chatbot built around a simple
  local RAG pipeline. Source code lives in `src/`, tests in `tests/`, corpus
  files in `data/`, scripts in `scripts/`, and generated FAISS artifacts in
  `indexes/`.

## High-level architecture
1. **Local document corpus**: A set of `.txt` files for a chosen domain.
2. **Indexing**: Embed with `sentence-transformers/all-MiniLM-L6-v2` and index in FAISS.
3. **Retrieval**: Fetch the most relevant excerpts for a user question.
4. **Generation**: Answer using retrieved context with a free/local model (e.g., `flan-t5-base` or `mistralai/Mistral-7B-Instruct` via text-generation-inference).
5. **Interface**: Simple terminal script or notebook. Optional: LangChain integration and history storage.

## Key conventions
- Use only local models or free Hugging Face endpoints; avoid paid APIs.
- Keep the pipeline simple: FAISS retrieval + MiniLM embeddings + a lightweight generator.
- Maintain a minimal terminal CLI unless the project scope expands.
- Keep unit tests lightweight and mocked; reserve real model downloads for
  `tests/e2e_smoke.py` or documented manual checks.
