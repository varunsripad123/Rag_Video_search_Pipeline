"""High-level pipeline orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from blake3 import blake3
from skimage.metrics import peak_signal_noise_ratio

from src.config import AppConfig
from src.core.chunking import chunk_dataset, persist_metadata
from src.core.embedding import EmbeddingExtractor
from src.core.indexing import build_index
from src.core.codecs import NeuralVideoCodec
from src.utils import track_stage
from src.utils.io import ManifestEntry
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames

LOGGER = get_logger(__name__)


def run_pipeline(config: AppConfig) -> Path:
    """Execute chunking, encoding, embedding extraction, and index build."""

    chunks = _chunk(config)
    manifest = _encode_and_embed(config, chunks)
    metadata_path = config.data.processed_dir / "metadata.json"
    persist_metadata(metadata_path, manifest)
    index_path = build_index(config, manifest)
    return index_path


def _chunk(config: AppConfig) -> List[VideoChunk]:
    with track_stage("chunking"):
        chunks = chunk_dataset(config)
    return chunks


def _encode_and_embed(config: AppConfig, chunks: List[VideoChunk]) -> List[ManifestEntry]:
    with track_stage("embedding"):
        extractor = EmbeddingExtractor(config)
        extractor.configure_precision()
        extractor.load()

    codec = NeuralVideoCodec(config.codec, device=config.models.device)

    embeddings_dir = config.data.processed_dir / "embeddings"
    tokens_dir = config.data.processed_dir / "tokens"
    sideinfo_dir = config.data.processed_dir / "sideinfo"
    for directory in (embeddings_dir, tokens_dir, sideinfo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest: List[ManifestEntry] = []
    for chunk in chunks:
        frames = load_video_frames(chunk.video_path)
        entry, _ = build_manifest_entry(
            config=config,
            codec=codec,
            extractor=extractor,
            chunk=chunk,
            frames=frames,
            embeddings_dir=embeddings_dir,
            tokens_dir=tokens_dir,
            sideinfo_dir=sideinfo_dir,
        )
        manifest.append(entry)

    return manifest


def build_manifest_entry(
    config: AppConfig,
    codec: NeuralVideoCodec,
    extractor: EmbeddingExtractor,
    chunk: VideoChunk,
    frames: np.ndarray,
    embeddings_dir: Path,
    tokens_dir: Path,
    sideinfo_dir: Path,
) -> tuple[ManifestEntry, np.ndarray]:
    """Encode embeddings and metadata for a single chunk."""

    token_path = tokens_dir / f"{chunk.video_path.stem}.npy"
    encoded = codec.encode(frames, token_path)

    decoded = codec.decode(token_path)
    quality = _quality_metrics(frames, decoded)

    sideinfo = {
        "codebook_id": config.codec.codebook_id,
        "layout": "tiles16x16",
        "dims": frames.shape[1:3],
        "time_stride": frames.shape[0],
    }
    sideinfo_path = sideinfo_dir / f"{chunk.video_path.stem}.json"
    sideinfo_path.write_text(json.dumps(sideinfo, indent=2), encoding="utf-8")

    embedding = extractor.encode_frames(frames)
    embedding_path = embeddings_dir / f"{chunk.video_path.stem}.npy"
    np.save(embedding_path, embedding)

    token_bytes = encoded.size_bytes
    source_bytes = Path(chunk.video_path).stat().st_size
    ratio = source_bytes / max(token_bytes, 1)
    digest = blake3(token_path.read_bytes()).hexdigest()

    entry = ManifestEntry(
        manifest_id=str(uuid4()),
        tenant_id=config.data.default_tenant,
        stream_id=_stream_id(chunk),
        label=chunk.label,
        chunk_path=str(chunk.video_path.resolve()),
        token_path=str(token_path.resolve()),
        sideinfo_path=str(sideinfo_path.resolve()),
        embedding_path=str(embedding_path.resolve()),
        start_time=chunk.start_frame / chunk.fps,
        end_time=chunk.end_frame / chunk.fps,
        fps=chunk.fps,
        codebook_id=config.codec.codebook_id,
        model_id=config.codec.model_id,
        byte_size=token_bytes,
        ratio=ratio,
        hash=f"blake3:{digest}",
        tags=[chunk.label],
        quality_stats=quality,
    )
    return entry, embedding


def _stream_id(chunk: VideoChunk) -> str:
    base = chunk.video_path.stem
    return base.split("_chunk")[0]


def _quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    min_frames = min(original.shape[0], reconstructed.shape[0])
    original = original[:min_frames]
    reconstructed = reconstructed[:min_frames]
    if reconstructed.size == 0:
        return {"psnr": 0.0, "vmaf": 0.0}
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255.0)
    mae = float(np.abs(original - reconstructed).mean())
    vmaf_proxy = max(0.0, min(100.0, 100.0 - mae / 255.0 * 100.0))
    return {"psnr": float(psnr), "vmaf": float(vmaf_proxy)}
