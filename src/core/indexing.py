"""FAISS indexing utilities."""
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
    """Wrapper around FAISS index for similarity search."""

    def __init__(self, dim: int, nlist: int, nprobe: int, use_gpu: bool = True):
        self.dim = dim
        description = f"IVF{nlist},Flat"
        self.index = faiss.index_factory(dim, description, faiss.METRIC_INNER_PRODUCT)
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.nprobe = nprobe

    def train(self, embeddings: np.ndarray) -> None:
        LOGGER.info("Training FAISS index", extra={"num_vectors": embeddings.shape[0]})
        faiss.normalize_L2(embeddings)
        self.index.train(embeddings)

    def add(self, embeddings: np.ndarray) -> None:
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        INDEX_SIZE.set(self.index.ntotal)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def save(self, path: Path) -> None:
        cpu_index = faiss.index_gpu_to_cpu(self.index) if isinstance(self.index, faiss.GpuIndex) else self.index
        faiss.write_index(cpu_index, str(path))

    def load(self, path: Path) -> None:
        index = faiss.read_index(str(path))
        self.index = index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        return distances, indices


def build_index(config: AppConfig, metadata: List[ManifestEntry]) -> Path:
    """Build FAISS index from embeddings stored on disk."""

    embeddings = []
    for item in metadata:
        embedding = np.load(item.embedding_path)
        embeddings.append(embedding)
    matrix = np.stack(embeddings)

    with track_stage("index_train"):
        index = FAISSIndex(
            dim=matrix.shape[1],
            nlist=config.index.nlist,
            nprobe=config.index.nprobe,
            use_gpu=config.index.use_gpu,
        )
        index.train(matrix)
        index.add(matrix)

    index_path = config.index.faiss_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    LOGGER.info("Saved FAISS index", extra={"path": str(index_path), "vectors": len(metadata)})
    return index_path
