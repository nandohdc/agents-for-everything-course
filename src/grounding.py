"""Lightweight checks for whether retrieved context can support a question."""

from __future__ import annotations

import re
from typing import Iterable

REFUSAL_ANSWER = "I cannot answer this based on the provided context."

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "best",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "keep",
    "me",
    "my",
    "of",
    "or",
    "should",
    "the",
    "their",
    "them",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _normalize_word(word: str) -> str:
    """Normalize a word enough for simple corpus/question term matching."""
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def significant_terms(text: str) -> list[str]:
    """Return ordered non-stopword terms from text."""
    terms: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        normalized = _normalize_word(word)
        if len(normalized) <= 2 or normalized in _STOPWORDS:
            continue
        if normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def context_supports_question(question: str, context_parts: Iterable[str]) -> bool:
    """Return whether the context contains all significant question terms.

    This intentionally conservative guard prevents generation when retrieval
    misses a key term, such as asking for an espresso roast recommendation when
    the returned context only discusses roast levels in general.
    """
    terms = significant_terms(question)
    context_text = " ".join(part for part in context_parts if part).strip()
    if not context_text:
        return False
    if not terms:
        return True

    context_terms = set(significant_terms(context_text))
    return all(term in context_terms for term in terms)
