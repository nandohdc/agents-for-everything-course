"""Module for building prompts from retrieved context and user queries."""

from typing import Any, Dict, List, Tuple


QA_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following retrieved context to answer the user's question.
Please answer from context only. If the answer cannot be found in the context, answer "I cannot answer this based on the provided context."
Do not use outside information.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(
    question: str,
    retrieved_chunks: List[Tuple[Dict[str, Any], float]]
) -> str:
    """Build a QA prompt using retrieved context.
    
    Args:
        question: The user's question.
        retrieved_chunks: A list of tuples containing chunk metadata and distance.
                          The metadata dict should have 'text' and 'source' keys.
                          
    Returns:
        The fully constructed prompt string ready to be passed to an LLM.
    """
    context_parts = []
    
    for i, (chunk, _) in enumerate(retrieved_chunks):
        text = chunk.get("text", "").strip()
        source = chunk.get("source", "Unknown")
        if text:
            context_parts.append(f"[{i + 1}] Source: {source}\n{text}\n")
            
    context_str = "\n".join(context_parts).strip()
    
    if not context_str:
        context_str = "No relevant context found."
        
    return QA_PROMPT_TEMPLATE.format(context=context_str, question=question)
