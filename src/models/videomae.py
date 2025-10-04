"""VideoMAE encoder wrapper."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEModel

from .base import BaseEncoder


LOGGER = logging.getLogger(__name__)


class VideoMAEEncoder(BaseEncoder):
    """Encodes video clips with VideoMAE."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def load(self) -> None:  # noqa: D401
        """Load the VideoMAE model and feature extractor."""

        try:
            self.model = VideoMAEModel.from_pretrained(self.model_name).to(self.device)
            self.processor = VideoMAEFeatureExtractor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VideoMAE model unavailable, using fallback encoder", exc_info=exc)
            self.model = None
            self.processor = None
            self._use_fallback = True

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        frames_list = list(frames)
        if self._use_fallback or self.model is None or self.processor is None:
            return self._fallback_encode(frames_list)
        inputs = self.processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._autocast():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings
