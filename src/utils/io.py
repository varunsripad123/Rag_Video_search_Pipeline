"""Manifest persistence helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(slots=True)
class ManifestRecord:
    """Structured metadata stored for each processed chunk."""

    manifest_id: str
    tenant_id: str
    stream_id: str
    label: str
    t0: str
    t1: str
    start_time: float
    end_time: float
    codebook_id: str
    model_id: str
    chunk_path: str
    token_uri: str
    sideinfo_uri: str
    embedding_path: str
    byte_size: int
    ratio: float
    hash: str
    quality_stats: dict[str, float]
    tags: List[str]


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    """Persist manifest records to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_manifest(path: Path) -> List[ManifestRecord]:
    """Load manifest records from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [ManifestRecord(**item) for item in data]


__all__ = ["ManifestRecord", "read_manifest", "write_manifest"]
