"""FAISS indexing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.config import AppConfig
from src.utils import INDEX_SIZE, track_stage
from src.utils.io import ManifestRecord
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class FAISSIndex:
    """Wrapper around FAISS index for similarity search."""

    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 8, use_gpu: bool = True):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.index = index

    def train(self, embeddings: np.ndarray) -> None:
        LOGGER.info("Training FAISS index", extra={"num_vectors": embeddings.shape[0]})
        faiss.normalize_L2(embeddings)
        self.index.train(embeddings)

    def add(self, embeddings: np.ndarray) -> None:
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        INDEX_SIZE.set(self.index.ntotal)

    def save(self, path: Path) -> None:
        index = self.index
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, str(path))

    @classmethod
    def load_from_path(cls, path: Path, use_gpu: bool = True, nprobe: int = 8) -> "FAISSIndex":
        index = faiss.read_index(str(path))
        instance = object.__new__(cls)
        instance.dim = index.d
        instance.nlist = getattr(index, "nlist", 1024)
        instance.nprobe = nprobe
        instance.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if instance.use_gpu:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            instance.index = index
        instance.index.nprobe = nprobe
        INDEX_SIZE.set(instance.index.ntotal)
        return instance

    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        return distances, indices


def build_index(config: AppConfig, manifest: List[ManifestRecord]) -> Path:
    """Build FAISS index from embeddings stored on disk."""

    if not manifest:
        raise RuntimeError("Manifest is empty; nothing to index.")
    embeddings = []
    for item in manifest:
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
    LOGGER.info("Saved FAISS index", extra={"path": str(index_path), "vectors": len(manifest)})
    return index_path
