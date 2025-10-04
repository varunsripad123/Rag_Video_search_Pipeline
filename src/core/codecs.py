"""Neural video codec components."""
from __future__ import annotations

import json
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


@dataclass(slots=True)
class EncodedArtifact:
    """Encoded chunk artifact."""

    token_path: Path
    sideinfo_path: Path
    latent_shape: Tuple[int, ...]
    entropy_scale: Iterable[float] | None


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

    def encode(self, frames: np.ndarray, output_dir: Path, stem: str) -> EncodedArtifact:
        """Encode frames and persist compressed representation."""

        output_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = output_dir / f"{stem}.npz"
        sideinfo_path = output_dir / f"{stem}.json"

        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        entropy_scale = None
        if self.motion is not None and tensor.shape[2] > 1:
            ref = tensor[:, :, :-1, :, :]
            target = tensor[:, :, 1:, :, :]
            ref_flat = ref.permute(0, 2, 1, 3, 4).reshape(-1, ref.shape[1], ref.shape[3], ref.shape[4])
            target_flat = target.permute(0, 2, 1, 3, 4).reshape(-1, target.shape[1], target.shape[3], target.shape[4])
            flow = self.motion(ref_flat, target_flat)
            LOGGER.debug("Motion flow magnitude", extra={"mean": float(flow.abs().mean())})

        reconstructed, latent = self.compressor(tensor)
        if self.bottleneck is not None:
            latent, scale = self.bottleneck(latent)
            entropy_scale = scale.detach().cpu().tolist()
            LOGGER.debug("Entropy bottleneck scale", extra={"scale": entropy_scale})

        latent_np = latent.detach().cpu().squeeze(0).numpy()
        np.savez_compressed(tokens_path, latent=latent_np)

        sideinfo = {"latent_shape": latent_np.shape, "entropy_scale": entropy_scale}
        sideinfo_path.write_text(json.dumps(sideinfo, indent=2), encoding="utf-8")

        return EncodedArtifact(
            token_path=tokens_path,
            sideinfo_path=sideinfo_path,
            latent_shape=latent_np.shape,
            entropy_scale=entropy_scale,
        )

    def decode(self, artifact: EncodedArtifact | Path) -> np.ndarray:
        """Decode latent representation back to video frames."""

        token_path = artifact.token_path if isinstance(artifact, EncodedArtifact) else artifact
        latent_data = np.load(token_path)
        latent = torch.from_numpy(latent_data["latent"]).unsqueeze(0).to(self.device)
        reconstruction = self.compressor.decoder(latent).squeeze(0).permute(1, 2, 3, 0)
        frames = (reconstruction.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return frames
