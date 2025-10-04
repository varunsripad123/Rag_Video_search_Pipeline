"""Retrieval logic using FAISS and metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.config import AppConfig
from src.core.indexing import FAISSIndex
from src.utils.io import ManifestEntry, read_manifest
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Search result entry."""

    manifest_id: str
    label: str
    score: float
    start_time: float
    end_time: float
    asset_url: str


class Retriever:
    """Performs embedding lookup and scoring."""

    def __init__(self, config: AppConfig, metadata_path: Path):
        self.config = config
        self.metadata_path = metadata_path
        self.metadata = read_manifest(metadata_path)
        if not self.metadata:
            raise RuntimeError("Metadata is empty. Run the pipeline first.")
        sample_embedding = np.load(self.metadata[0].embedding_path)
        dim = int(sample_embedding.shape[0])
        self.index = FAISSIndex(
            dim=dim,
            nlist=config.index.nlist,
            nprobe=config.index.nprobe,
            use_gpu=config.index.use_gpu,
        )
        self.index.load(config.index.faiss_index_path)
        self.dim = dim

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k=top_k)
        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievalResult(
                    manifest_id=meta.manifest_id,
                    label=meta.label,
                    score=float(score),
                    start_time=meta.start_time,
                    end_time=meta.end_time,
                    asset_url=meta.chunk_path,
                )
            )
        return results

    def add_entry(self, entry: ManifestEntry, embedding: np.ndarray) -> None:
        """Add a manifest entry and embedding into the index at runtime."""

        if embedding.shape[-1] != self.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[-1]} does not match index ({self.dim})")
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(entry)


def expand_query(query_embedding: np.ndarray, history: Sequence[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Blend query embedding with conversation history."""

    if not history:
        return query_embedding
    history_stack = np.stack(history)
    blended = (1 - alpha) * query_embedding + alpha * history_stack.mean(axis=0)
    return blended
