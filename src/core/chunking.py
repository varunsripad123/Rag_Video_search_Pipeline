from pathlib import Path
from typing import Iterator, List
import json
from src.config import AppConfig
from src.utils.video import iter_videos, chunk_video
from src.utils.io import ManifestEntry

def chunk_dataset(cfg: AppConfig) -> Iterator:
    for path,label in iter_videos(Path(cfg.data.root_dir)):
        yield from chunk_video(
            path, label,
            Path(cfg.data.processed_dir)/"chunks",
            cfg.data.chunk_duration, cfg.data.min_frames, int(cfg.data.frame_rate * 15)
        )

def persist_metadata(path: Path, manifest: List[ManifestEntry]) -> None:
    """Write manifest entries to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in manifest], handle, indent=2)
