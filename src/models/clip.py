"""CLIP encoder wrapper."""
from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .base import BaseEncoder


LOGGER = logging.getLogger(__name__)


class CLIPEncoder(BaseEncoder):
    """Encodes video keyframes using CLIP."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def load(self) -> None:  # noqa: D401
        """Load the CLIP model and processor."""

        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CLIP model unavailable, using fallback encoder", exc_info=exc)
            self.model = None
            self.processor = None
            self._use_fallback = True

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        if self._use_fallback or self.model is None or self.processor is None:
            return self._fallback_encode(list(frames))
        pil_images: List[np.ndarray] = [frame[:, :, ::-1] for frame in frames]
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._autocast():
            outputs = self.model.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings
