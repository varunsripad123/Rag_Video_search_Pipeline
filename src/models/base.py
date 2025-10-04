"""Base encoder utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch


class BaseEncoder(ABC):
    """Common utilities for encoder models."""

    def __init__(self, device: str = "cuda", precision: str = "fp16", quantize: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.quantize = quantize
        self.model = None
        self.processor = None
        self._use_fallback = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def encode(self, inputs: Iterable) -> np.ndarray:
        """Return embedding vectors for inputs."""

    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "fp16":
            return torch.cuda.amp.autocast()
        return torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=False)

    def _fallback_encode(self, frames: Iterable[np.ndarray], components: int = 512) -> np.ndarray:
        """Compute lightweight histogram-based embeddings when models are unavailable."""

        arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
        if not arrays:
            raise ValueError("No frames provided to fallback encoder")
        stacked = np.stack(arrays)
        bins = max(16, components // 3)
        histograms = []
        for channel in range(min(stacked.shape[-1], 3)):
            channel_values = stacked[..., channel].ravel()
            hist, _ = np.histogram(channel_values, bins=bins, range=(0, 255), density=True)
            histograms.append(hist)
        embedding = np.concatenate(histograms)
        if embedding.size < components:
            embedding = np.pad(embedding, (0, components - embedding.size))
        elif embedding.size > components:
            embedding = embedding[:components]
        embedding = embedding.reshape(1, -1)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        return embedding
