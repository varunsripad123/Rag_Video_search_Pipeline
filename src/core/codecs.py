"""Neural video codec components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import CodecSettings
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MotionEstimator(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([ref, tgt], dim=1))


class EntropyBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, C, T, H, W) or (B, C, H, W)
        # Reshape scale to broadcast correctly
        if x.ndim == 5:  # Video tensor (B, C, T, H, W)
            scale_view = self.scale.view(1, -1, 1, 1, 1)
        else:  # Image tensor (B, C, H, W)
            scale_view = self.scale.view(1, -1, 1, 1)
        
        noise = torch.randn_like(x) * torch.sigmoid(scale_view)
        return x + noise, self.scale


class ResidualCompressor(nn.Module):
    def __init__(self, res_ch: int, lat_ch: int):
        super().__init__()
        # Use 2D convolutions to process each frame independently
        self.encoder = nn.Sequential(
            nn.Conv2d(3, res_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(res_ch, lat_ch, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lat_ch, res_ch, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(res_ch, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Reshape to process all frames at once: (B*T, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        # Encode
        latent_2d = self.encoder(x_2d)  # (B*T, lat_ch, H, W)
        
        # Decode
        recon_2d = self.decoder(latent_2d)  # (B*T, 3, H, W)
        
        # Reshape back to video format
        _, lat_ch, H_lat, W_lat = latent_2d.shape
        latent = latent_2d.reshape(B, T, lat_ch, H_lat, W_lat).permute(0, 2, 1, 3, 4)  # (B, lat_ch, T, H, W)
        recon = recon_2d.reshape(B, T, 3, H, W).permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        
        return recon, latent


@dataclass
class EncodedChunk:
    path: Path
    size_bytes: int


class NeuralVideoCodec:
    def __init__(self, settings: CodecSettings, device: str = "cuda"):
        self.settings = settings
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.motion = MotionEstimator().to(self.device) if settings.enable_motion_estimation else None
        self.compressor = ResidualCompressor(settings.residual_channels, settings.latent_channels).to(self.device)
        self.bottleneck = EntropyBottleneck(settings.latent_channels).to(self.device) if settings.entropy_bottleneck else None

    def encode_tensor(self, tensor: torch.Tensor, output_path: Path) -> EncodedChunk:
        """
        Encode a tensor of shape (1,3,T,H,W) directly.
        Simplified version that just saves a downsampled version.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simply save the tensor as compressed numpy
        # Convert to uint8 to save space
        x_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        np.savez_compressed(output_path, data=x_np)
        
        return EncodedChunk(path=output_path, size_bytes=output_path.stat().st_size)

    def decode(self, path: Path) -> np.ndarray:
        # Load compressed data
        data = np.load(path)
        x_np = data['data']  # (1, 3, T, H, W) uint8
        
        # Convert back to (T, H, W, 3)
        x_np = x_np.squeeze(0).transpose(1, 2, 3, 0)  # (T, H, W, 3)
        return x_np
