"""Answer generation module."""
from __future__ import annotations

from typing import Sequence, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Cache model and tokenizer to avoid reloading every time
_cached_models = {}

def generate_answer(
    prompt: str,
    context: Sequence[str],
    model_name: str = "microsoft/phi-2",
    device: Optional[str] = None,  # Changed to Optional[str]
) -> str:
    """Generate a textual answer given retrieved context."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_name not in _cached_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)
            model.eval()
            _cached_models[model_name] = (tokenizer, model)
        else:
            tokenizer, model = _cached_models[model_name]

        system_prompt = "You are an expert video search assistant. Use the context to answer succinctly."
        context_block = "\n".join(context)
        full_prompt = f"{system_prompt}\nContext:\n{context_block}\nQuestion: {prompt}\nAnswer:"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()

    except Exception as exc:
        LOGGER.exception("Falling back to template generation", exc_info=exc)
        return f"Based on the retrieved segments, here is what I found: {'; '.join(context)}"
