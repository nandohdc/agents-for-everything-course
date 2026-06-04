"""Unit tests for the Generator class."""

import unittest
from unittest.mock import MagicMock, patch

from src.generator import Generator


class TestGenerator(unittest.TestCase):

    @patch("src.generator.AutoModelForSeq2SeqLM")
    @patch("src.generator.AutoTokenizer")
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model
        
        # Setup mock returns
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer_instance
        
        self.mock_model_instance = MagicMock()
        self.mock_model.from_pretrained.return_value.to.return_value = self.mock_model_instance
        
        self.generator = Generator(model_name="fake-model")
        
    def test_initialization(self):
        """Test that the generator initializes correctly."""
        self.mock_tokenizer.from_pretrained.assert_called_once_with("fake-model")
        self.mock_model.from_pretrained.assert_called_once_with("fake-model")
        self.assertEqual(self.generator.tokenizer, self.mock_tokenizer_instance)
        self.assertEqual(self.generator.model, self.mock_model_instance)

    def test_generate_empty_prompt(self):
        """Test generating with an empty prompt returns empty string."""
        result = self.generator.generate("   ")
        self.assertEqual(result, "")
        self.mock_model_instance.generate.assert_not_called()

    def test_generate_with_sampling(self):
        """Test text generation with sampling (temperature > 0)."""
        # Setup mock behavior
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        self.mock_tokenizer_instance.return_value.to.return_value = mock_inputs
        self.mock_model_instance.generate.return_value = [MagicMock()]
        self.mock_tokenizer_instance.decode.return_value = "Mocked answer"

        # Execute
        result = self.generator.generate("What is AI?", max_new_tokens=50, temperature=0.7)

        # Assertions
        self.assertEqual(result, "Mocked answer")
        self.mock_tokenizer_instance.assert_called_once_with("What is AI?", return_tensors="pt")
        self.mock_model_instance.generate.assert_called_once_with(
            **mock_inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        self.mock_tokenizer_instance.decode.assert_called_once_with(
            self.mock_model_instance.generate.return_value[0],
            skip_special_tokens=True
        )

    def test_generate_greedy(self):
        """Test text generation with greedy decoding (temperature = 0)."""
        # Setup mock behavior
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        self.mock_tokenizer_instance.return_value.to.return_value = mock_inputs
        self.mock_model_instance.generate.return_value = [MagicMock()]
        self.mock_tokenizer_instance.decode.return_value = "Greedy answer"

        # Execute
        result = self.generator.generate("What is AI?", max_new_tokens=100, temperature=0.0)

        # Assertions
        self.assertEqual(result, "Greedy answer")
        self.mock_model_instance.generate.assert_called_once_with(
            **mock_inputs,
            max_new_tokens=100,
            do_sample=False
        )


if __name__ == "__main__":
    unittest.main()
