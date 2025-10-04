"""Video Swin Transformer encoder."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import VideoMAEFeatureExtractor, VideoSwinModel

from .base import BaseEncoder


class VideoSwinEncoder(BaseEncoder):
    """Encodes video clips with Video Swin transformer."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def load(self) -> None:  # noqa: D401
        """Load the Video Swin transformer and feature extractor."""

        try:
            self.model = VideoSwinModel.from_pretrained(self.model_name).to(self.device)
            self.processor = VideoMAEFeatureExtractor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - exercised in production environments
            self._mark_fallback(exc)

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        if self._fallback or self.model is None or self.processor is None:
            return self._fallback_encode(frames, target_dim=512)
        frames_list = list(frames)
        inputs = self.processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._autocast():
            outputs = self.model(**inputs)
        embeddings = outputs.pooler_output.cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings
