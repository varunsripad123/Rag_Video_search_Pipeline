"""FAISS indexing utilities for CPU-only use."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np

from src.config import AppConfig
from src.utils import INDEX_SIZE, track_stage
from src.utils.io import ManifestEntry
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class FAISSIndex:
    """Wrapper around FAISS index for similarity search in CPU-only mode."""

    def __init__(self, dim: int, nlist: int, nprobe: int, use_gpu: bool = False):
        # Ignore GPU options for CPU-only
        self.dim = dim
        description = f"IVF{nlist},Flat"
        self.index = faiss.index_factory(dim, description, faiss.METRIC_INNER_PRODUCT)
        # Remove GPU code entirely
        # if use_gpu and faiss.get_num_gpus() > 0:
        #     res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.nprobe = nprobe

    def train(self, embeddings: np.ndarray) -> None:
        LOGGER.info("Training FAISS index", extra={"num_vectors": embeddings.shape[0]})
        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.train(embeddings)

    def add(self, embeddings: np.ndarray) -> None:
        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        INDEX_SIZE.set(self.index.ntotal)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        index = faiss.read_index(str(path))
        self.index = index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure float32 for FAISS
        queries = queries.astype(np.float32)
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        return distances, indices


def build_index(config: AppConfig, metadata: List[ManifestEntry]) -> Path:
    """Build FAISS index from embeddings stored on disk with CPU-only support."""

    embeddings = []
    shapes = {}
    
    for item in metadata:
        # Handle both string and Path objects
        emb_path = Path(item.embedding_path) if isinstance(item.embedding_path, str) else item.embedding_path
        embedding = np.load(emb_path)
        
        # Ensure embedding is 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # Track shapes for debugging
        shape_key = embedding.shape[0]
        if shape_key not in shapes:
            shapes[shape_key] = []
        shapes[shape_key].append(str(item.embedding_path))
        
        embeddings.append(embedding)
    
    # Log shape distribution
    LOGGER.info(f"Embedding shape distribution: {dict((k, len(v)) for k, v in shapes.items())}")
    
    # If we have multiple shapes, we need to handle it
    if len(shapes) > 1:
        LOGGER.warning(f"Found embeddings with different shapes: {list(shapes.keys())}")
        # Find the most common shape
        most_common_shape = max(shapes.keys(), key=lambda k: len(shapes[k]))
        LOGGER.info(f"Using most common shape: {most_common_shape}")
        
        # Filter to only use embeddings with the most common shape
        filtered_embeddings = []
        filtered_metadata = []
        for emb, item in zip(embeddings, metadata):
            if emb.shape[0] == most_common_shape:
                filtered_embeddings.append(emb)
                filtered_metadata.append(item)
            else:
                LOGGER.warning(f"Skipping {item.embedding_path} with shape {emb.shape}")
        
        embeddings = filtered_embeddings
        metadata = filtered_metadata
    
    matrix = np.stack(embeddings)

    with track_stage("index_train"):
        index = FAISSIndex(
            dim=matrix.shape[1],
            nlist=config.index.nlist,
            nprobe=config.index.nprobe,
            use_gpu=False  # explicitly disable GPU
        )
        index.train(matrix)
        index.add(matrix)

    index_path = config.index.faiss_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    LOGGER.info("Saved FAISS index", extra={"path": str(index_path), "vectors": len(metadata)})
    return index_path
