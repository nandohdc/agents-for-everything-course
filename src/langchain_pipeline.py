"""Optional LangChain implementation of the RAG pipeline (bonus, Issue #14).

This is an *alternative* backend to the baseline pipeline, exposed through the
CLI via ``--engine langchain``. It is deliberately self-contained and imports
LangChain **lazily** (inside ``__init__``) so the baseline pipeline never
depends on LangChain being installed — importing this module is cheap and only
constructing :class:`LangChainRAG` pulls in the extras.

Install the optional extras first::

    pip install -r requirements-langchain.txt
"""

from typing import List, Tuple, Union
from pathlib import Path

from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.generator import Generator
from src.grounding import REFUSAL_ANSWER, context_supports_question

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "google/flan-t5-base"

# Mirrors src/prompt_builder.QA_PROMPT_TEMPLATE so both backends share the same
# "answer from context only" guidance.
LANGCHAIN_QA_TEMPLATE = """You are a helpful assistant. Use the following retrieved context to answer the user's question.
Please answer from context only. If the answer cannot be found in the context, answer "I cannot answer this based on the provided context."
Do not use outside information.

Context:
{context}

Question: {question}
Answer:"""


class LangChainRAG:
    """A LangChain-backed retrieval + generation pipeline over the local corpus.

    Builds an in-memory FAISS vector store from the same ``data/`` corpus the
    baseline uses (chunked identically), embeds with ``all-MiniLM-L6-v2`` and
    generates with the same local flan-t5 generator as the baseline pipeline.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        k: int = 3,
        max_tokens: int = 128,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        # Lazy imports keep LangChain an optional dependency.
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document as LCDocument
            from langchain_core.prompts import PromptTemplate
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "LangChain extras are not installed. Install them with: "
                "pip install -r requirements-langchain.txt"
            ) from exc

        self.k = k
        self.max_tokens = max_tokens

        # 1. Load + chunk the same corpus the baseline pipeline uses.
        docs = load_documents(data_dir)
        if not docs:
            raise ValueError(f"No documents found in '{data_dir}'.")

        chunks = chunk_documents(
            docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunks:
            raise ValueError(f"No chunks produced from documents in '{data_dir}'.")

        lc_docs = [
            LCDocument(
                page_content=chunk.text,
                metadata={
                    "source": chunk.metadata.get("source_path")
                    or chunk.metadata.get("source_filename", "Unknown")
                },
            )
            for chunk in chunks
        ]

        # 2. Build the FAISS vector store with MiniLM embeddings.
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_documents(lc_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # 3. Use the baseline local generator so Transformers task support stays
        # consistent between the two CLI engines.
        self.generator = Generator(model_name=model_name)
        self.prompt = PromptTemplate.from_template(LANGCHAIN_QA_TEMPLATE)

    def answer(self, question: str) -> Tuple[str, List[str]]:
        """Answer a question end-to-end; returns ``(answer, source_labels)``."""
        docs = self.retriever.invoke(question)
        if not context_supports_question(question, (doc.page_content for doc in docs)):
            sources: List[str] = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            return REFUSAL_ANSWER, sources

        context = "\n\n".join(
            f"[{i + 1}] Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        prompt_text = self.prompt.format(
            context=context or "No relevant context found.", question=question
        )

        answer = self.generator.generate(
            prompt_text, max_new_tokens=self.max_tokens, temperature=0.0
        )

        sources: List[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)

        return answer.strip(), sources
