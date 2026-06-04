"""Optional question/answer history storage for the RAG CLI.

Persists each interaction as a single JSON line (JSONL) so a session can be
inspected later. The feature is optional (toggled from the CLI) and is
intentionally dependency-free — it uses only the standard-library ``json``
module, in line with the project's lightweight-RAG scope.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union


def append_interaction(
    history_file: Union[str, Path],
    question: str,
    answer: str,
    sources: List[str],
) -> None:
    """Append a single question/answer interaction to a JSONL history file.

    Creates the parent directory (and the file) on first write. Each call
    appends exactly one JSON object on its own line with the keys
    ``timestamp`` (UTC, ISO 8601), ``question``, ``answer`` and ``sources``.

    Args:
        history_file: Path to the JSONL history file.
        question: The user's question.
        answer: The generated answer.
        sources: Source identifiers for the retrieved context (may be empty).
    """
    history_file = Path(history_file)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": list(sources),
    }

    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
