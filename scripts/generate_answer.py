"""Smoke test script to manually verify text generation."""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` can be imported when
# running the script directly (e.g. ``python scripts/generate_answer.py``).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.generator import Generator  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description="Test local LLM generation.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="The prompt to send to the model."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature."
    )
    args = parser.parse_args()

    print("Loading generator (this might download the model on first run)...")
    generator = Generator(model_name="google/flan-t5-base")
    
    print(f"\nPrompt: {args.prompt}")
    print("Generating answer...")
    
    answer = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
