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
    auto_labels: dict = None  # Auto-generated labels (objects, actions, captions, audio)


class Retriever:
    """Performs embedding lookup and scoring."""

    def __init__(self, config: AppConfig, metadata_path: Path):
        self.config = config
        self.metadata_path = metadata_path
        self.metadata = read_manifest(metadata_path)
        if not self.metadata:
            raise RuntimeError("Metadata is empty. Run the pipeline first.")
        # Handle both string and Path objects
        emb_path = Path(self.metadata[0].embedding_path) if isinstance(self.metadata[0].embedding_path, str) else self.metadata[0].embedding_path
        sample_embedding = np.load(emb_path)
        dim = int(sample_embedding.shape[0])
        self.index = FAISSIndex(
            dim=dim,
            nlist=config.index.nlist,
            nprobe=config.index.nprobe,
            use_gpu=config.index.use_gpu,
        )
        self.index.load(config.index.faiss_index_path)
        self.dim = dim

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filter_objects: List[str] = None,
        filter_action: str = None,
        min_confidence: float = 0.0
    ) -> List[RetrievalResult]:
        """Search with optional auto-label filters.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_objects: Only return results containing these objects (e.g., ['person', 'car'])
            filter_action: Only return results with this action (e.g., 'walking')
            min_confidence: Minimum auto-label confidence score
        
        Returns:
            List of retrieval results
        """
        query_embedding = query_embedding.reshape(1, -1)
        # Search more results initially if filtering
        search_k = top_k * 3 if (filter_objects or filter_action or min_confidence > 0) else top_k
        scores, indices = self.index.search(query_embedding, k=search_k)
        
        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            
            # Apply auto-label filters
            if meta.auto_labels:
                # Filter by objects
                if filter_objects:
                    detected_objects = meta.auto_labels.get('objects', [])
                    if not any(obj in detected_objects for obj in filter_objects):
                        continue
                
                # Filter by action
                if filter_action:
                    detected_action = meta.auto_labels.get('action', '').lower()
                    if filter_action.lower() not in detected_action:
                        continue
                
                # Filter by confidence
                if min_confidence > 0:
                    confidence = meta.auto_labels.get('confidence', 0.0)
                    if confidence < min_confidence:
                        continue
            elif filter_objects or filter_action or min_confidence > 0:
                # Skip results without auto_labels if filters are specified
                continue
            
            results.append(
                RetrievalResult(
                    manifest_id=meta.manifest_id,
                    label=meta.label,
                    score=float(score),
                    start_time=meta.start_time,
                    end_time=meta.end_time,
                    asset_url=meta.chunk_path,
                    auto_labels=meta.auto_labels  # Include auto-labels in results
                )
            )
            
            # Stop once we have enough filtered results
            if len(results) >= top_k:
                break
        
        return results[:top_k]

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
