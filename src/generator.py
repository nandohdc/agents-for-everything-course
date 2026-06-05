"""Local LLM generation module using Hugging Face transformers."""

from typing import Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Generator:
    """Generates answers from prompts using a local Sequence-to-Sequence model."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        hf_token: Union[str, None] = None,
    ):
        """Initialize the generator with a local model.

        Args:
            model_name: The Hugging Face model hub identifier.
            hf_token: Optional Hugging Face user access token for model downloads.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Support Apple Silicon (MPS) if available
        if not torch.cuda.is_available() and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"

        kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(self.device)

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        temperature: float = 0.7
    ) -> str:
        """Generate text based on a prompt.

        Args:
            prompt: The input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Higher values make output more random,
                         values close to 0 make it more deterministic.

        Returns:
            The generated text.
        """
        if not prompt.strip():
            return ""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        do_sample = temperature > 0.0
        
        # When temperature is 0, do_sample must be False and temperature is ignored.
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
        }
        
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
