"""Video processing utilities."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Generator, Tuple

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class VideoChunk:
    video_path: Path
    label: str
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps

def iter_videos(root_dir: Path) -> Generator[Tuple[Path,str], None, None]:
    for label_dir in root_dir.iterdir():
        if not label_dir.is_dir(): continue
        for video_file in label_dir.glob("*.mp4"):
            yield video_file, label_dir.name

def chunk_video(
    video_path: Path, label: str, output_dir: Path,
    chunk_duration: float, min_frames: int, target_fps: int
) -> Iterable[VideoChunk]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_chunk = max(int(chunk_duration*fps), min_frames)
    idx, start, buf = 0, 0, []
    ok, frame = cap.read()
    while ok:
        buf.append(frame)
        if len(buf)==per_chunk or cap.get(cv2.CAP_PROP_POS_FRAMES)==total:
            out = output_dir/f"{video_path.stem}_chunk{idx:04d}.mp4"
            write_video(out, buf, fps)
            yield VideoChunk(out,label,start,start+len(buf),fps)
            start+=len(buf); idx+=1; buf=[]
        ok, frame = cap.read()
    cap.release()

def probe_fps(video_path: Path) -> float:
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0  # Default to 30 if unable to detect

def write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> None:
    frames = list(frames)
    h,w,_ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wri = cv2.VideoWriter(str(path), fourcc, fps, (w,h))
    for f in frames: wri.write(f)
    wri.release()

def load_video_frames(video_path: Path, max_frames: int = 16, max_size: int = 128) -> torch.Tensor:
    """
    Load a video and return as tensor with shape (1,3,T,H,W).
    Limits to max_frames and downsamples to max_size to avoid memory issues.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames=[]
    ok,frame=cap.read()
    while ok and len(frames) < max_frames:
        # Downsample if needed
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        frames.append(frame)
        ok,frame=cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames: {video_path}")
    arr=np.stack(frames)  # (T,H,W,3)
    LOGGER.info(f"[load_video_frames] raw shape {arr.shape}")

    # Convert to tensor: (T,H,W,3) -> (1,3,T,H,W)
    data = arr.transpose(0, 3, 1, 2)  # (T,3,H,W)
    tensor = torch.from_numpy(data).unsqueeze(0).float() / 255.0  # (1,T,3,H,W)
    tensor = tensor.permute(0, 2, 1, 3, 4)  # (1,3,T,H,W)
    LOGGER.info(f"[load_video_frames] out shape {tensor.shape}")
    return tensor
