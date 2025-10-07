"""CLIP-only embedding extractor for fixing search scores."""

import numpy as np
import torch
from pathlib import Path
from typing import Iterable, List, Optional

from src.config import AppConfig
from src.models import CLIPEncoder
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames

LOGGER = get_logger(__name__)


class CLIPEmbeddingExtractor:
    """CLIP-only embedding extractor for fixing search scores."""

    def __init__(self, config: AppConfig):
        self.config = config
        device = config.models.device
        precision = config.models.precision
        quantize = config.models.quantize

        self.clip = CLIPEncoder(
            config.models.clip_model_name, device=device, precision=precision, quantize=quantize
        )

    def load(self) -> None:
        """Load CLIP model."""
        LOGGER.info("Loading CLIP model for CLIP-only embeddings")
        try:
            self.clip.load()
            LOGGER.info("✅ CLIP model loaded successfully!")
        except Exception as e:
            LOGGER.error(f"❌ Failed to load CLIP model: {e}")
            raise

    def encode_frames(self, frames) -> np.ndarray:
        """Encode frames using only CLIP."""
        try:
            # Convert tensor to numpy if needed
            if isinstance(frames, torch.Tensor):
                frames = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                frames = (frames * 255).astype(np.uint8)

            # Prepare frames for CLIP
            frame_list = self._prepare_frames_for_clip(frames)

            if not frame_list:
                LOGGER.error("No valid frames for CLIP encoding")
                return np.zeros((512,), dtype=np.float32)

            # Encode with CLIP only
            try:
                clip_embedding = self.clip.encode(frame_list)
                clip_embedding = self._normalize_embedding(clip_embedding, "CLIP", expected_dim=512)
                LOGGER.debug(f"CLIP embedding shape: {clip_embedding.shape}")
                return clip_embedding.astype(np.float32)
            except Exception as e:
                LOGGER.error(f"CLIP encoding failed: {e}")
                return np.zeros((512,), dtype=np.float32)

        except Exception as e:
            LOGGER.error(f"Frame encoding failed: {e}", exc_info=True)
            return np.zeros((512,), dtype=np.float32)

    def _prepare_frames_for_clip(self, frames: np.ndarray) -> List[np.ndarray]:
        """Convert frames for CLIP encoder."""
        if frames.ndim == 4:
            frame_list = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame_list.append(frame)
            return frame_list
        else:
            LOGGER.warning(f"Unexpected frames format: {frames.shape}")
            return []

    def _normalize_embedding(self, embedding: np.ndarray, encoder_name: str, expected_dim: int = None) -> np.ndarray:
        """Normalize embedding to ensure consistent shape."""
        try:
            if embedding is None:
                LOGGER.warning(f"{encoder_name} returned None embedding")
                fallback_dim = expected_dim or 512
                return np.zeros((fallback_dim,), dtype=np.float32)

            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                embedding = embedding.reshape(embedding.shape[0], -1)

            if embedding.shape[0] > 1:
                embedding = embedding.mean(axis=0, keepdims=True)

            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            if expected_dim is not None and embedding.shape[0] != expected_dim:
                if embedding.shape[0] < expected_dim:
                    padding = np.zeros((expected_dim - embedding.shape[0],), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:expected_dim]

            return embedding.astype(np.float32)

        except Exception as e:
            LOGGER.error(f"Error normalizing {encoder_name} embedding: {e}")
            fallback_dim = expected_dim or 512
            return np.zeros((fallback_dim,), dtype=np.float32)

    def encode_chunk(self, chunk: VideoChunk) -> Optional[np.ndarray]:
        """Encode a single video chunk using only CLIP."""
        try:
            LOGGER.debug("Encoding chunk with CLIP only", extra={"chunk": str(chunk.video_path)})
            frames = load_video_frames(chunk.video_path)
            return self.encode_frames(frames)
        except Exception as e:
            LOGGER.error(f"CLIP encoding failed for chunk {chunk.video_path}: {e}", exc_info=True)
            return None

    def encode_chunks(self, chunks: Iterable[VideoChunk], output_dir: Path) -> List[Path]:
        """Encode multiple video chunks and save embeddings."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        total_chunks = 0
        successful_chunks = 0

        for chunk in chunks:
            total_chunks += 1
            embedding = self.encode_chunk(chunk)

            if embedding is None:
                LOGGER.warning(f"Skipping chunk {chunk.video_path.stem} - CLIP encoding failed")
                continue

            try:
                path = output_dir / f"{chunk.video_path.stem}.npy"
                np.save(path, embedding)
                saved_paths.append(path)
                successful_chunks += 1
                LOGGER.debug(f"Saved CLIP embedding for {chunk.video_path.stem} to {path}")
            except Exception as e:
                LOGGER.error(f"Failed to save CLIP embedding for {chunk.video_path.stem}: {e}")

        LOGGER.info(f"✅ Successfully encoded {successful_chunks}/{total_chunks} chunks with CLIP only")
        return saved_paths
