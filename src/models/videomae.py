"""VideoMAE encoder wrapper with robust frame handling."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor
from src.utils.logging import get_logger

# Defer transformers import to avoid import errors
VideoMAEModel = None
HFVideoMAEFeatureExtractor = None

LOGGER = get_logger(__name__)


def _try_import_transformers():
    """Try to import transformers components at runtime."""
    global VideoMAEModel, HFVideoMAEFeatureExtractor
    if VideoMAEModel is not None:
        return True
    
    try:
        from transformers import VideoMAEModel as _VideoMAEModel
        VideoMAEModel = _VideoMAEModel
        try:
            from transformers import VideoMAEImageProcessor
            HFVideoMAEFeatureExtractor = VideoMAEImageProcessor
        except ImportError:
            from transformers import VideoMAEFeatureExtractor
            HFVideoMAEFeatureExtractor = VideoMAEFeatureExtractor
        return True
    except Exception as e:
        LOGGER.warning(f"Could not import transformers VideoMAE: {e}")
        return False


class BaseEncoder:
    """Base class for device and precision management."""

    def __init__(
        self,
        device: str = "cuda",
        precision: str = "fp32",
        quantize: bool = False,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA not available, switching to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.precision = precision
        self.quantize = quantize

    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "fp16":
            return torch.cuda.amp.autocast()
        return nullcontext()


class VideoMAEEncoder(BaseEncoder):
    """Encoder for VideoMAE with inline feature extraction."""

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        device: str = "cuda",
        precision: str = "fp32",
        quantize: bool = False,
    ):
        super().__init__(device=device, precision=precision, quantize=quantize)
        self.model_name = model_name
        self.model = None
        self._use_fallback = False
        self.feature_extractor = None

    def load(self) -> None:
        # Try to import transformers at runtime
        if not _try_import_transformers():
            LOGGER.warning("VideoMAE components not available, using fallback")
            self._use_fallback = True
            return
        
        # Load feature extractor
        try:
            self.feature_extractor = HFVideoMAEFeatureExtractor.from_pretrained(self.model_name)
        except Exception as e:
            LOGGER.warning(f"Could not load VideoMAE feature extractor: {e}")
            self.feature_extractor = None
            self._use_fallback = True
            return
        
        # Load model
        try:
            model = VideoMAEModel.from_pretrained(self.model_name)
            self.model = model.to(self.device)
            self.model.eval()
        except Exception as e:
            LOGGER.error(f"Failed to load VideoMAE model: {e}", exc_info=True)
            self.model = None
            self._use_fallback = True

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        frames_list = list(frames)
        if not frames_list:
            LOGGER.warning("No frames provided to encode.")
            return self._fallback_encode()

        # Check if we should use fallback BEFORE trying to use feature_extractor
        if self._use_fallback or self.model is None or self.feature_extractor is None:
            return self._fallback_encode()

        # Ensure each frame is HWC RGB and dtype uint8
        processed = []
        for f in frames_list:
            if f.ndim != 3 or f.shape[2] not in (3, 4):
                raise ValueError(f"Frame shape must be HWC with 3 or 4 channels, got {f.shape}")
            img = f[..., :3]  # drop alpha if present
            processed.append(img)

        # Use HuggingFace feature extractor
        try:
            inputs = self.feature_extractor(
                images=processed,
                return_tensors="pt",
            )
            pixel_values: Tensor = inputs.pixel_values.to(self.device)  # shape (batch, T, C, H, W)
        except Exception as e:
            LOGGER.error(f"Feature extraction failed: {e}", exc_info=True)
            return self._fallback_encode()

        try:
            with self._autocast():
                outputs = self.model(pixel_values)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("Model output missing last_hidden_state")
            # hidden shape: (batch, seq_len, hidden_dim)
            emb = hidden.mean(dim=1)  # (batch, hidden_dim)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            emb_np = emb[0].cpu().numpy()
            return emb_np
        except Exception as e:
            LOGGER.error(f"Encoding failed: {e}", exc_info=True)
            return self._fallback_encode()

    def _fallback_encode(self) -> np.ndarray:
        dim = getattr(self.model.config, "hidden_size", 768) if self.model else 768
        LOGGER.warning(f"Fallback: returning zero vector of dim {dim}")
        return np.zeros(dim, dtype=np.float32)
