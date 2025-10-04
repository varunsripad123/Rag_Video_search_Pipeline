"""Base encoder utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class BaseEncoder(ABC):
    """Common utilities for encoder models."""

    def __init__(self, device: str = "cuda", precision: str = "fp16", quantize: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.quantize = quantize
        self.model = None
        self.processor = None
        self._fallback = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def encode(self, inputs: Iterable) -> np.ndarray:
        """Return embedding vectors for inputs."""

    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "fp16":
            return torch.cuda.amp.autocast()
        if self.device.type == "cpu":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float32)

    def _fallback_encode(self, frames: Iterable[np.ndarray], target_dim: int = 512) -> np.ndarray:
        """Generate a deterministic statistical embedding when models are unavailable."""

        array = np.asarray(list(frames), dtype=np.float32)
        if array.size == 0:
            raise ValueError("Received no frames to encode")
        array = array / 255.0
        stats = [array.mean(axis=(0, 1, 2)), array.std(axis=(0, 1, 2))]
        histograms = []
        for channel in range(array.shape[-1]):
            hist, _ = np.histogram(array[..., channel], bins=128, range=(0.0, 1.0), density=True)
            histograms.append(hist)
        features = np.concatenate([*stats, *histograms])
        features = features / (np.linalg.norm(features) + 1e-8)
        if features.size < target_dim:
            features = np.pad(features, (0, target_dim - features.size))
        else:
            features = features[:target_dim]
        return features.astype(np.float32)[None, :]

    def _mark_fallback(self, reason: Exception) -> None:
        self._fallback = True
        LOGGER.warning(
            "Falling back to statistical embeddings", extra={"encoder": self.__class__.__name__, "error": str(reason)}
        )
