import unittest

from src.prompt_builder import QA_PROMPT_TEMPLATE, build_prompt


class TestPromptBuilder(unittest.TestCase):
    def test_build_prompt_with_context(self):
        question = "What is the capital of France?"
        chunks = [
            ({"text": "Paris is the capital of France.", "source": "geo.txt"}, 0.1),
            ({"text": "France is in Europe.", "source": "europe.txt"}, 0.5)
        ]
        
        prompt = build_prompt(question, chunks)
        
        self.assertIn("Question: What is the capital of France?", prompt)
        self.assertIn("[1] Source: geo.txt", prompt)
        self.assertIn("Paris is the capital of France.", prompt)
        self.assertIn("[2] Source: europe.txt", prompt)
        self.assertIn("France is in Europe.", prompt)
        self.assertIn("answer from context only", prompt.lower())
        
    def test_build_prompt_empty_context(self):
        question = "What is the capital of France?"
        chunks = []
        
        prompt = build_prompt(question, chunks)
        
        self.assertIn("Question: What is the capital of France?", prompt)
        self.assertIn("No relevant context found.", prompt)
        
    def test_missing_keys_in_chunk(self):
        question = "Test?"
        chunks = [
            ({"text": "Just some text"}, 0.1),
            ({"source": "only_source.txt"}, 0.2)
        ]
        
        prompt = build_prompt(question, chunks)
        
        self.assertIn("[1] Source: Unknown", prompt)
        self.assertIn("Just some text", prompt)
        self.assertNotIn("only_source.txt", prompt)

if __name__ == "__main__":
    unittest.main()
