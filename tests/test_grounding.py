import unittest
from unittest.mock import MagicMock

from src.cli import answer_question
from src.grounding import (
    REFUSAL_ANSWER,
    context_supports_question,
    significant_terms,
)


class StubEngine:
    def __init__(self, results):
        self.results = results

    def query(self, question, k=3):
        return self.results


class TestGrounding(unittest.TestCase):
    def test_significant_terms_ignore_question_words(self):
        terms = significant_terms("What roast level is best for espresso?")

        self.assertEqual(terms, ["roast", "level", "espresso"])

    def test_context_supports_storage_question(self):
        context = [
            "How should I store coffee beans? Store coffee in an airtight "
            "container away from heat, light, moisture, and strong odors so "
            "they stay fresh."
        ]

        self.assertTrue(
            context_supports_question(
                "How should I store coffee beans to keep them fresh?", context
            )
        )

    def test_context_rejects_unsupported_espresso_roast_question(self):
        context = [
            "Light roast coffee usually keeps more of the bean's original "
            "character. Medium roast balances sweetness and body. Dark roast "
            "has heavier body and roast-driven flavor."
        ]

        self.assertFalse(
            context_supports_question(
                "What roast level is best for espresso?", context
            )
        )

    def test_answer_question_refuses_before_generation_when_context_unsupported(self):
        results = [
            (
                {
                    "source": "coffee_beans_and_roast_levels.txt",
                    "text": (
                        "Light roast coffee usually keeps more of the bean's "
                        "original character. Medium roast balances sweetness "
                        "and body. Dark roast has heavier body."
                    ),
                },
                1.0,
            )
        ]
        generator = MagicMock()
        generator.generate.return_value = "Light"

        answer, returned_results = answer_question(
            "What roast level is best for espresso?",
            StubEngine(results),
            generator,
            k=3,
            max_tokens=32,
        )

        self.assertEqual(answer, REFUSAL_ANSWER)
        self.assertEqual(returned_results, results)
        generator.generate.assert_not_called()

    def test_answer_question_generates_when_context_has_question_terms(self):
        results = [
            (
                {
                    "source": "coffee_storage_and_freshness.txt",
                    "text": (
                        "How should I store coffee beans? Store coffee in an "
                        "airtight container away from heat, light, moisture, "
                        "and strong odors so it stays fresh."
                    ),
                },
                0.5,
            )
        ]
        generator = MagicMock()
        generator.generate.return_value = "Store beans in an airtight container."

        answer, returned_results = answer_question(
            "How should I store coffee beans to keep them fresh?",
            StubEngine(results),
            generator,
            k=3,
            max_tokens=32,
        )

        self.assertEqual(answer, "Store beans in an airtight container.")
        self.assertEqual(returned_results, results)
        generator.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
