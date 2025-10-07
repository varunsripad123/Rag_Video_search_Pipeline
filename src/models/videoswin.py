"""Video Swin encoder wrapper."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

from .base import BaseEncoder

LOGGER = logging.getLogger(__name__)

class VideoSwinEncoder(BaseEncoder):
    """Encodes video clips using Torchvision Swin3D transformer, extracting features."""

    def __init__(self, model_name: str = "swin_t", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._use_fallback = False

    def load(self) -> None:
        """Load the Torchvision Swin3D model with pretrained weights."""
        try:
            if self.model_name == "swin_t":
                weights = Swin3D_T_Weights.DEFAULT
                self.model = swin3d_t(weights=weights)
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")

            self.model = self.model.to(self.device)
            self.model.eval()
            self._use_fallback = False
        except Exception as exc:
            LOGGER.warning("Torchvision Video Swin model unavailable, using fallback encoder", exc_info=exc)
            self.model = None
            self._use_fallback = True

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        frames_list = list(frames)

        # Strip alpha channel if exists
        frames_list = [
            frame[:, :, :3] if frame.shape[2] == 4 else frame
            for frame in frames_list
        ]

        if self._use_fallback or self.model is None:
            return self._fallback_encode(frames_list).squeeze(0)

        # Prepare input: (T, H, W, C) -> (1, C, T, H, W) normalized to [0,1]
        video = np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        try:
            with autocast(enabled=True):
                # Extract features from the model's feature layers
                # Remove the classification head and get features
                x = video
                # Pass through the model's feature extraction layers
                for name, module in self.model.named_children():
                    if name == 'head':  # Skip the classification head
                        break
                    x = module(x)
                
                # Global average pooling to get a fixed-size embedding
                if x.dim() == 5:  # (B, C, T, H, W)
                    embeddings = torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
                elif x.dim() == 3:  # (B, T, C) - already pooled
                    embeddings = x.mean(dim=1)
                else:
                    embeddings = x.flatten(1)

            embeddings = embeddings.cpu().numpy()
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            return embeddings.squeeze(0)  # Return 1D array
        except Exception as e:
            LOGGER.error(f"VideoSwin encoding failed: {e}", exc_info=True)
            return self._fallback_encode(frames_list).squeeze(0)
