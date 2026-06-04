# Copilot instructions

## Build, test, lint
- No build/test/lint commands are defined in this repository.

## Project overview
- This repository currently contains the final project spec in `final-project.md` for building an FAQ chatbot using a simple RAG pipeline.

## High-level architecture
1. **Local document corpus**: A set of `.txt` files for a chosen domain.
2. **Indexing**: Embed with `sentence-transformers/all-MiniLM-L6-v2` and index in FAISS.
3. **Retrieval**: Fetch the most relevant excerpts for a user question.
4. **Generation**: Answer using retrieved context with a free/local model (e.g., `flan-t5-base` or `mistralai/Mistral-7B-Instruct` via text-generation-inference).
5. **Interface**: Simple terminal script or notebook. Optional: LangChain integration and history storage.

## Key conventions
- Use only local models or free Hugging Face endpoints; avoid paid APIs.
- Keep the pipeline simple: FAISS retrieval + MiniLM embeddings + a lightweight generator.
- Maintain a minimal user interface (CLI or notebook) unless the project scope expands.
