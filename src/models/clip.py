"""CLIP encoder wrapper."""
from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import torch

from .base import BaseEncoder

LOGGER = logging.getLogger(__name__)

# Defer transformers import to avoid import errors
CLIPModel = None
CLIPProcessor = None


def _try_import_clip():
    """Try to import CLIP components at runtime."""
    global CLIPModel, CLIPProcessor
    if CLIPModel is not None:
        return True
    
    try:
        from transformers import CLIPModel as _CLIPModel, CLIPProcessor as _CLIPProcessor
        CLIPModel = _CLIPModel
        CLIPProcessor = _CLIPProcessor
        return True
    except Exception as e:
        LOGGER.warning(f"Could not import transformers CLIP: {e}")
        return False

class CLIPEncoder(BaseEncoder):
    """Encodes video keyframes using CLIP."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._use_fallback = False

    def load(self) -> None:
        """Load the CLIP model and processor."""
        # Try to import transformers at runtime
        if not _try_import_clip():
            LOGGER.warning("CLIP components not available, using fallback")
            return
        
        try:
            import os
            
            # Get HF token from environment
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Try to load from cache first (offline mode) to avoid rate limits
            try:
                result = CLIPModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    ignore_mismatched_sizes=True,
                    token=token,
                    local_files_only=True  # Use cache only
                )
                self.model = result.to(self.device)
                self.model.eval()
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_name,
                    token=token,
                    local_files_only=True  # Use cache only
                )
                LOGGER.info("✅ Loaded CLIP from cache (offline mode)")
                self._use_fallback = False
                return
            except Exception as cache_error:
                LOGGER.info(f"Cache not available, downloading: {cache_error}")
            
            # If cache fails, download with token
            result = CLIPModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
                ignore_mismatched_sizes=True,
                token=token
            )
            self.model = result.to(self.device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                token=token
            )
            self._use_fallback = False
        except Exception as e:
            LOGGER.warning("CLIP model unavailable, using fallback encoder", exc_info=e)
            self.model = None
            self.processor = None
            self._use_fallback = True

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        frames_list = list(frames)

        # Strip alpha channel if present, ensure RGB format
        frames_list = [
            frame[:, :, :3] if frame.shape[2] == 4 else frame
            for frame in frames_list
        ]

        if self._use_fallback or self.model is None or self.processor is None:
            return self._fallback_encode(frames_list)

        # Convert BGR (OpenCV) to RGB by reversing last channel dim
        pil_images: List[np.ndarray] = [frame[:, :, ::-1] for frame in frames_list]

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._autocast():
            outputs = self.model.get_image_features(**inputs)

        embeddings = outputs.cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings
