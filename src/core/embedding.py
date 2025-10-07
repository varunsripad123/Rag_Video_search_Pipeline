"""Embedding extraction for video chunks - Fixed version."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from src.config import AppConfig
from src.models import CLIPEncoder, VideoMAEEncoder, VideoSwinEncoder
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames

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
        """Load all embedding models."""
        LOGGER.info("Loading embedding models")
        for encoder_name, encoder in [("CLIP", self.clip), ("VideoMAE", self.videomae), ("VideoSwin", self.videoswin)]:
            try:
                encoder.load()
                LOGGER.info(f"Successfully loaded {encoder_name} encoder")
            except Exception as e:
                LOGGER.error(f"Failed to load {encoder_name} encoder: {e}")

    def encode_chunk(self, chunk: VideoChunk) -> Optional[np.ndarray]:
        """Encode a single video chunk."""
        try:
            LOGGER.debug("Encoding chunk", extra={"chunk": str(chunk.video_path)})
            frames = load_video_frames(chunk.video_path)
            return self.encode_frames(frames)
        except Exception as e:
            LOGGER.error(f"Encoding failed for chunk {chunk.video_path}: {e}", exc_info=True)
            return None

    def _prepare_frames_for_encoders(self, frames: np.ndarray) -> List[np.ndarray]:
        """
        Convert frames from 4D array (T, H, W, C) to list of 3D frames [(H, W, C), ...].
        Strip alpha channel if present.
        """
        # If frames is 4D (T, H, W, C), convert to list of 3D frames
        if frames.ndim == 4:
            frame_list = []
            for i in range(frames.shape[0]):
                frame = frames[i]  # (H, W, C)
                # Strip alpha channel if present
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame_list.append(frame)
            return frame_list
        else:
            # If frames is already a list or different format, handle accordingly
            LOGGER.warning(f"Unexpected frames format: {frames.shape if hasattr(frames, 'shape') else type(frames)}")
            return []

    def _normalize_embedding(self, embedding: np.ndarray, encoder_name: str, expected_dim: int = None) -> np.ndarray:
        """Normalize embedding to ensure consistent shape and format."""
        try:
            if embedding is None:
                LOGGER.warning(f"{encoder_name} returned None embedding")
                fallback_dim = expected_dim or 512
                return np.zeros((fallback_dim,), dtype=np.float32)
            
            # Ensure embedding is 2D (batch_size, features)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                # Flatten extra dimensions
                embedding = embedding.reshape(embedding.shape[0], -1)
            
            # If batch size > 1, take the mean across batch dimension
            if embedding.shape[0] > 1:
                embedding = embedding.mean(axis=0, keepdims=True)
            
            # Ensure we have a single embedding vector
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)  # Shape: (features,)
            
            # Pad or truncate to expected dimension if specified
            if expected_dim is not None and embedding.shape[0] != expected_dim:
                if embedding.shape[0] < expected_dim:
                    # Pad with zeros
                    padding = np.zeros((expected_dim - embedding.shape[0],), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
                else:
                    # Truncate
                    embedding = embedding[:expected_dim]
                LOGGER.debug(f"{encoder_name} embedding adjusted to {expected_dim} dimensions")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            LOGGER.error(f"Error normalizing {encoder_name} embedding: {e}")
            fallback_dim = expected_dim or 512
            return np.zeros((fallback_dim,), dtype=np.float32)

    def encode_frames(self, frames) -> np.ndarray:
        """Encode frames using all three encoders and concatenate results.
        
        Args:
            frames: Either a torch.Tensor of shape (1,3,T,H,W) or numpy array of shape (T,H,W,C)
        """
        try:
            # Convert tensor to numpy if needed
            if isinstance(frames, torch.Tensor):
                # (1,3,T,H,W) -> (T,H,W,3)
                frames = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                frames = (frames * 255).astype(np.uint8)
            
            # Prepare frames for encoders
            frame_list = self._prepare_frames_for_encoders(frames)
            
            if not frame_list:
                LOGGER.error("No valid frames to encode")
                # Return a fallback embedding
                return np.zeros((768 * 3,), dtype=np.float32)  # 3 encoders × 768 features

            LOGGER.debug(f"Encoding {len(frame_list)} frames")

            # Encode with each encoder
            embeddings = {}
            
            # CLIP encoding (expected: 512 dims)
            try:
                clip_embedding = self.clip.encode(frame_list)
                clip_embedding = self._normalize_embedding(clip_embedding, "CLIP", expected_dim=512)
                embeddings['clip'] = clip_embedding
                LOGGER.debug(f"CLIP embedding shape: {clip_embedding.shape}")
            except Exception as e:
                LOGGER.error(f"CLIP encoding failed: {e}")
                embeddings['clip'] = np.zeros((512,), dtype=np.float32)

            # VideoMAE encoding (expected: 768 dims)
            try:
                mae_embedding = self.videomae.encode(frame_list)
                mae_embedding = self._normalize_embedding(mae_embedding, "VideoMAE", expected_dim=768)
                embeddings['videomae'] = mae_embedding
                LOGGER.debug(f"VideoMAE embedding shape: {mae_embedding.shape}")
            except Exception as e:
                LOGGER.error(f"VideoMAE encoding failed: {e}")
                embeddings['videomae'] = np.zeros((768,), dtype=np.float32)

            # VideoSwin encoding (expected: 8 dims based on current output)
            try:
                swin_embedding = self.videoswin.encode(frame_list)
                # Don't specify expected_dim yet, let's see what we get
                swin_embedding = self._normalize_embedding(swin_embedding, "VideoSwin")
                embeddings['videoswin'] = swin_embedding
                LOGGER.debug(f"VideoSwin embedding shape: {swin_embedding.shape}")
            except Exception as e:
                LOGGER.error(f"VideoSwin encoding failed: {e}")
                embeddings['videoswin'] = np.zeros((512,), dtype=np.float32)

            # Log embedding shapes for debugging
            shapes = {name: emb.shape for name, emb in embeddings.items()}
            LOGGER.debug(f"Individual embedding shapes: {shapes}")

            # Ensure all embeddings are 1D and same dtype
            normalized_embeddings = []
            for name, emb in embeddings.items():
                if emb.ndim != 1:
                    LOGGER.warning(f"{name} embedding is not 1D: {emb.shape}")
                    emb = emb.flatten()
                normalized_embeddings.append(emb)

            # Concatenate embeddings
            try:
                final_embedding = np.concatenate(normalized_embeddings, axis=0)
                LOGGER.debug(f"Final concatenated embedding shape: {final_embedding.shape}")
                return final_embedding.astype(np.float32)
            except Exception as e:
                LOGGER.error(f"Failed to concatenate embeddings: {e}")
                # Return a fallback embedding
                total_dim = sum(emb.shape[0] for emb in normalized_embeddings)
                return np.zeros((total_dim,), dtype=np.float32)

        except Exception as e:
            LOGGER.error(f"Frame encoding failed: {e}", exc_info=True)
            # Return a fallback embedding
            return np.zeros((768 * 3,), dtype=np.float32)

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
                LOGGER.warning(f"Skipping chunk {chunk.video_path.stem} - encoding failed")
                continue
            
            try:
                path = output_dir / f"{chunk.video_path.stem}.npy"
                np.save(path, embedding)
                saved_paths.append(path)
                successful_chunks += 1
                LOGGER.debug(f"Saved embedding for {chunk.video_path.stem} to {path}")
            except Exception as e:
                LOGGER.error(f"Failed to save embedding for {chunk.video_path.stem}: {e}")

        LOGGER.info(f"Successfully encoded {successful_chunks}/{total_chunks} chunks")
        return saved_paths

    @staticmethod
    def configure_precision() -> None:
        """Configure PyTorch for optimized precision."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            LOGGER.info("Enabled CUDNN benchmark for better performance")
