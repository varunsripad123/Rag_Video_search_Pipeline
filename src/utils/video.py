"""Video processing utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Tuple

import cv2
import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class VideoChunk:
    """Metadata about a processed chunk."""

    video_path: Path
    label: str
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


def iter_videos(root_dir: Path) -> Generator[Tuple[Path, str], None, None]:
    """Yield (path, label) pairs from a labeled directory tree."""

    for label_dir in root_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for video_file in label_dir.glob("*.mp4"):
            yield video_file, label_dir.name


def chunk_video(
    video_path: Path,
    label: str,
    output_dir: Path,
    chunk_duration: float,
    min_frames: int,
    target_fps: int,
) -> Iterable[VideoChunk]:
    """Split a video into uniform chunks and store them as individual files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_chunk = max(int(math.floor(chunk_duration * fps)), min_frames)

    chunk_index = 0
    success, frame = capture.read()
    buffer = []
    current_start = 0

    while success:
        buffer.append(frame)
        if len(buffer) == frames_per_chunk or capture.get(cv2.CAP_PROP_POS_FRAMES) == frame_count:
            chunk_path = output_dir / f"{video_path.stem}_chunk{chunk_index:04d}.mp4"
            write_video(chunk_path, buffer, fps)
            chunk = VideoChunk(
                video_path=chunk_path,
                label=label,
                start_frame=current_start,
                end_frame=current_start + len(buffer),
                fps=fps,
            )
            LOGGER.debug("Created chunk", extra={"chunk": chunk.__dict__})
            yield chunk
            current_start += len(buffer)
            chunk_index += 1
            buffer = []
        success, frame = capture.read()

    capture.release()


def write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> None:
    """Persist a list of frames to disk as an mp4 file."""

    frames = list(frames)
    if not frames:
        raise ValueError("Cannot write empty video")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_video_frames(video_path: Path) -> np.ndarray:
    """Load a video file into a numpy array of frames (T, H, W, C)."""

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    success, frame = capture.read()
    while success:
        frames.append(frame)
        success, frame = capture.read()
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return np.stack(frames)


def probe_fps(video_path: Path, default: float = 24.0) -> float:
    """Return the frames-per-second of a video file."""

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps or fps <= 0:
        return default
    return float(fps)
