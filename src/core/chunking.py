"""Chunking pipeline for videos."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.config import AppConfig
from src.utils import video
from src.utils.io import ManifestEntry, write_manifest
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def chunk_dataset(config: AppConfig) -> List[video.VideoChunk]:
    """Process all videos into chunks and return chunk descriptors."""

    root = config.data.root_dir
    output_dir = config.data.processed_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[video.VideoChunk] = []
    for video_path, label in video.iter_videos(root):
        LOGGER.info("Chunking video", extra={"video": str(video_path), "label": label})
        for chunk in video.chunk_video(
            video_path=video_path,
            label=label,
            output_dir=output_dir,
            chunk_duration=config.data.chunk_duration,
            min_frames=config.data.min_frames,
            target_fps=config.data.frame_rate,
        ):
            metadata.append(chunk)
    return metadata


def persist_metadata(path: Path, metadata: Iterable[ManifestEntry]) -> None:
    """Persist manifest entries to disk."""

    write_manifest(path, metadata)
