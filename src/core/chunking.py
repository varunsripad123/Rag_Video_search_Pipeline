"""Chunking pipeline for videos."""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.config import AppConfig
from src.utils import video
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def chunk_dataset(config: AppConfig) -> List[video.VideoChunk]:
    """Process all videos into chunks and return chunk descriptors."""

    root = config.data.root_dir
    if not root.exists():
        message = f"Dataset directory '{root}' does not exist."
        LOGGER.error(message)
        raise FileNotFoundError(message)
    output_dir = config.data.processed_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[video.VideoChunk] = []
    for video_path, label in video.iter_videos(root):
        LOGGER.info("Chunking video", extra={"video": str(video_path), "label": label})
        try:
            for chunk in video.chunk_video(
                video_path=video_path,
                label=label,
                output_dir=output_dir,
                chunk_duration=config.data.chunk_duration,
                min_frames=config.data.min_frames,
                target_fps=config.data.frame_rate,
            ):
                chunks.append(chunk)
        except Exception as exc:  # pragma: no cover - defensive logging for data issues
            LOGGER.warning(
                "Skipping video due to processing error",
                extra={"video": str(video_path), "error": str(exc)},
            )
    return chunks
