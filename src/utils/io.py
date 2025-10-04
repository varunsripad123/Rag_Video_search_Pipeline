"""I/O helpers for reading and writing metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List


@dataclass
class ManifestEntry:
    """Metadata persisted for each stored chunk/token window."""

    manifest_id: str
    tenant_id: str
    stream_id: str
    label: str
    chunk_path: str
    token_path: str
    sideinfo_path: str
    embedding_path: str
    start_time: float
    end_time: float
    fps: float
    codebook_id: str
    model_id: str
    byte_size: int
    ratio: float
    hash: str
    tags: list[str]
    quality_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestEntry":
        return cls(**data)


def write_manifest(path: Path, metadata: Iterable[ManifestEntry]) -> None:
    """Write manifest entries to disk as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in metadata], handle, indent=2)


def read_manifest(path: Path) -> List[ManifestEntry]:
    """Load manifest entries from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [ManifestEntry.from_dict(item) for item in data]


# Backwards compatibility helpers -------------------------------------------------


def write_metadata(path: Path, metadata: Iterable[ManifestEntry]) -> None:
    """Alias to :func:`write_manifest` for legacy callers."""

    write_manifest(path, metadata)


def read_metadata(path: Path) -> List[ManifestEntry]:
    """Alias to :func:`read_manifest` for legacy callers."""

    return read_manifest(path)


__all__ = ["ManifestEntry", "write_manifest", "read_manifest", "write_metadata", "read_metadata"]
