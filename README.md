# Agents for Everything — Coffee FAQ Chatbot

A terminal FAQ chatbot over a local coffee-domain corpus, built as a lightweight
**RAG** (Retrieval-Augmented Generation) pipeline. It uses only free/local
models — no paid APIs.

**Pipeline:** load `.txt` docs from `data/` → chunk → embed with
`sentence-transformers/all-MiniLM-L6-v2` → index with **FAISS** → retrieve the
most relevant chunks for a question → generate an answer with
`google/flan-t5-base`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency set has been validated with **Python 3.14**. The first run
downloads the MiniLM and flan-t5 models from Hugging Face (a few hundred MB),
which are then cached locally.

## Usage

### Ask a question (single-shot)

```bash
python -m src.cli "How should I store coffee beans to keep them fresh?"
```

Expected output (answer wording depends on the model):

```
Answer: Airtight container.

Sources:
  - data/coffee_storage_and_freshness.txt
```

If the FAISS index does not exist yet, the CLI **builds it automatically** from
`data/` before answering.

### Interactive session

Run without a question to start a REPL (type `exit` or `quit` to leave):

```bash
python -m src.cli
```

```
Coffee FAQ chatbot. Ask a question, or type 'exit'/'quit' to leave.

You: What roast level is best for espresso?

Answer: ...

Sources:
  - data/coffee_beans_and_roast_levels.txt

You: exit
Goodbye!
```

### Useful flags

| Flag | Default | Description |
|---|---|---|
| `-k`, `--top-k` | `3` | Number of chunks to retrieve. |
| `--index-file` | `indexes/faiss.index` | Path to the FAISS index. |
| `--metadata-dir` | `indexes` | Directory holding `metadata.json`. |
| `--data-dir` | `data` | Corpus used when (re)building the index. |
| `--rebuild` | off | Rebuild the index from `--data-dir` before answering. |
| `--model` | `google/flan-t5-base` | Generator model. |
| `--max-tokens` | `128` | Max new tokens to generate. |
| `--engine` | `baseline` | `baseline` or `langchain` (see below). |
| `--hf-token` | unset | Hugging Face token for authenticated model downloads. |
| `--history-file` | `history/qa_history.jsonl` | Where Q&A history is appended. |
| `--no-history` | off | Disable Q&A history storage. |

Prefer `HF_TOKEN` or `hf auth login` for regular use because command-line
tokens can be visible in shell history or process listings. Use `--hf-token`
only for short-lived local or CI commands:

```bash
python -m src.cli "How should I store coffee beans?" --hf-token "$HF_TOKEN"
```

### Building the index explicitly

The CLI auto-builds the index, but you can also build it (or its parts)
directly via the scripts in `scripts/`:

```bash
python -m src.cli --rebuild "..."        # rebuild via the CLI, then answer
# or, step by step:
python scripts/generate_embeddings.py    # data/ -> indexes/embeddings.npy + metadata.json
python scripts/build_index.py            # embeddings -> indexes/faiss.index
python scripts/query.py "..."            # retrieval only (no generation)
```

### Q&A history (optional, bonus — Issue #15)

Each interaction is appended as one JSON line to `history/qa_history.jsonl`
unless `--no-history` is passed:

```json
{"timestamp": "...", "question": "...", "answer": "...", "sources": ["data/..."]}
```

The `history/` directory is git-ignored. History errors never interrupt
answering.

### LangChain mode (optional, bonus — Issue #14)

An alternative backend built on LangChain is available. It is **not** required
for the baseline pipeline; install the optional extras first:

```bash
pip install -r requirements-langchain.txt
python -m src.cli --engine langchain "How should I store coffee beans?"
```

It builds an in-memory FAISS store from the same corpus, embeds with MiniLM and
uses LangChain retriever and prompt helpers before generating with the same
local flan-t5 model as the baseline pipeline. The baseline pipeline keeps
working without these extras installed (LangChain is imported lazily), and
missing optional dependencies are reported as concise CLI errors.

## Testing

Tests use the standard-library `unittest` module (pytest is not a dependency):

```bash
python -m unittest discover -s tests            # full unit suite (fast, mocked)
python -m unittest tests.test_history            # a single module
```

End-to-end smoke test — runs the **real** pipeline (indexing → retrieval →
generation) against the corpus and downloads the models on first run. It is a
standalone script, intentionally excluded from `unittest discover`:

```bash
python tests/e2e_smoke.py
```

It prints the retrieved sources and the generated answer, and exits `0` on
success (a non-empty answer was produced) or `1` on failure.

Run tests with the **repo root as the working directory**.

## Project layout

```
data/                 local coffee corpus (.txt)
indexes/              generated FAISS index + embeddings + metadata (build output)
history/              Q&A history (git-ignored, runtime output)
src/                  document_loader, chunker, embeddings, vector_store,
                      query_engine, prompt_builder, generator, cli,
                      history, langchain_pipeline
scripts/              runnable entry points (load/generate/build/query/answer)
tests/                unit tests + e2e_smoke.py
```
