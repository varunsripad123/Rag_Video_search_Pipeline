"""Neural video codec components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import CodecSettings
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MotionEstimator(nn.Module):
    """Simple convolutional motion estimator network."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, ref: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([ref, target], dim=1)
        return self.encoder(x)


class EntropyBottleneck(nn.Module):
    """Entropy bottleneck placeholder."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        noise = torch.randn_like(x) * torch.sigmoid(self.scale)
        return x + noise, self.scale


class ResidualCompressor(nn.Module):
    """Residual encoder/decoder."""

    def __init__(self, residual_channels: int, latent_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, residual_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(residual_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, residual_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(residual_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


@dataclass
class EncodedChunk:
    """Encoded chunk artifact."""

    path: Path
    size_bytes: int


class NeuralVideoCodec:
    """Codec orchestrating motion estimation and entropy bottlenecks."""

    def __init__(self, settings: CodecSettings, device: str = "cuda"):
        self.settings = settings
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.motion = MotionEstimator().to(self.device) if settings.enable_motion_estimation else None
        self.compressor = ResidualCompressor(settings.residual_channels, settings.latent_channels).to(self.device)
        self.bottleneck = (
            EntropyBottleneck(settings.latent_channels).to(self.device)
            if settings.entropy_bottleneck
            else None
        )

    def encode(self, frames: np.ndarray, output_path: Path) -> EncodedChunk:
        """Encode frames and persist compressed representation."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        if self.motion is not None:
            ref = tensor[:, :, :-1]
            target = tensor[:, :, 1:]
            flow = self.motion(ref.reshape(-1, *ref.shape[2:]), target.reshape(-1, *target.shape[2:]))
            LOGGER.debug("Motion flow magnitude", extra={"mean": float(flow.abs().mean())})

        reconstructed, latent = self.compressor(tensor)
        if self.bottleneck is not None:
            latent, scale = self.bottleneck(latent)
            LOGGER.debug("Entropy bottleneck scale", extra={"scale": scale.detach().cpu().tolist()})

        np.save(output_path, latent.detach().cpu().numpy())
        size_bytes = output_path.stat().st_size
        return EncodedChunk(path=output_path, size_bytes=size_bytes)

    def decode(self, encoded_path: Path) -> np.ndarray:
        """Decode latent representation back to video frames."""

        latent = torch.from_numpy(np.load(encoded_path)).to(self.device)
        reconstruction = self.compressor.decoder(latent).squeeze(0).permute(0, 2, 3, 1)
        return (reconstruction.detach().cpu().numpy() * 255.0).astype(np.uint8)
