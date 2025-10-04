"""Embedding extraction for video chunks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from src.config import AppConfig
from src.models import CLIPEncoder, VideoMAEEncoder, VideoSwinEncoder
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_frames

LOGGER = get_logger(__name__)


class EmbeddingExtractor:
    """Orchestrates multiple encoders and aggregates embeddings."""

    def __init__(self, config: AppConfig):
        self.config = config
        device = config.models.device
        precision = config.models.precision
        quantize = config.models.quantize
        self.clip = CLIPEncoder(
            config.models.clip_model_name, device=device, precision=precision, quantize=quantize
        )
        self.videomae = VideoMAEEncoder(
            config.models.videomae_model_name, device=device, precision=precision, quantize=quantize
        )
        self.videoswin = VideoSwinEncoder(
            config.models.videoswin_model_name, device=device, precision=precision, quantize=quantize
        )

    def load(self) -> None:
        LOGGER.info("Loading embedding models")
        for encoder in (self.clip, self.videomae, self.videoswin):
            encoder.load()

    def encode_chunk(self, chunk: VideoChunk) -> np.ndarray:
        LOGGER.debug("Encoding chunk", extra={"chunk": str(chunk.video_path)})
        frames = load_frames(chunk.video_path)
        return self.encode_frames(frames)

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        frame_sequence = [frame for frame in frames]
        clip_embedding = self.clip.encode(frame_sequence)
        mae_embedding = self.videomae.encode(frame_sequence)
        swin_embedding = self.videoswin.encode(frame_sequence)
        clip_pooled = clip_embedding.mean(axis=0)
        mae_pooled = mae_embedding.mean(axis=0)
        swin_pooled = swin_embedding.mean(axis=0)
        combined = np.concatenate([clip_pooled, mae_pooled, swin_pooled])
        return combined.astype(np.float32)

    def encode_chunks(self, chunks: Iterable[VideoChunk], output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        for chunk in chunks:
            embedding = self.encode_chunk(chunk)
            path = output_dir / f"{chunk.video_path.stem}.npy"
            np.save(path, embedding)
            saved_paths.append(path)
        return saved_paths

    @staticmethod
    def _load_frames(video_path: Path) -> np.ndarray:
        return load_frames(video_path)

    @staticmethod
    def configure_precision() -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
