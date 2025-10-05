"""High-level pipeline orchestration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from blake3 import blake3
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.config import AppConfig
from src.core.chunking import chunk_dataset
from src.core.codecs import EncodedArtifact, NeuralVideoCodec
from src.core.embedding import EmbeddingExtractor
from src.core.indexing import build_index
from src.utils import track_stage
from src.utils.io import ManifestRecord, write_manifest
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_frames

LOGGER = get_logger(__name__)


def run_pipeline(config: AppConfig) -> Path:
    """Execute chunking, compression, embedding extraction, and index build."""

    chunks = _chunk(config)
    if not chunks:
        raise RuntimeError(
            "No video chunks were produced. Ensure the dataset contains supported video files with enough frames."
        )
    manifest = _process_chunks(config, chunks)
    manifest_path = config.data.processed_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    index_path = build_index(config, manifest)
    return index_path


def _chunk(config: AppConfig) -> List[VideoChunk]:
    with track_stage("chunking"):
        chunks = chunk_dataset(config)
    return chunks


def _process_chunks(config: AppConfig, chunks: List[VideoChunk]) -> List[ManifestRecord]:
    with track_stage("processing"):
        extractor = EmbeddingExtractor(config)
        extractor.configure_precision()
        extractor.load()
        codec = NeuralVideoCodec(config.codec, device=config.models.device)

        embeddings_dir = config.data.processed_dir / "embeddings"
        tokens_dir = config.data.processed_dir / "tokens"
        sideinfo_dir = config.data.processed_dir / "sideinfo"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        tokens_dir.mkdir(parents=True, exist_ok=True)
        sideinfo_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[ManifestRecord] = []
        origin = datetime.now(timezone.utc)

        for chunk in chunks:
            frames = load_frames(chunk.video_path)
            embedding = extractor.encode_frames(frames)
            embedding_path = embeddings_dir / f"{chunk.video_path.stem}.npy"
            np.save(embedding_path, embedding)

            artifact = codec.encode(frames, tokens_dir, chunk.video_path.stem)
            sideinfo_target = sideinfo_dir / artifact.sideinfo_path.name
            if artifact.sideinfo_path != sideinfo_target:
                new_path = artifact.sideinfo_path.replace(sideinfo_target)
                artifact = EncodedArtifact(
                    token_path=artifact.token_path,
                    sideinfo_path=new_path,
                    latent_shape=artifact.latent_shape,
                    entropy_scale=artifact.entropy_scale,
                )

            decoded = codec.decode(artifact)
            quality = _compute_quality_metrics(frames, decoded)

            original_size = chunk.video_path.stat().st_size
            encoded_size = artifact.token_path.stat().st_size + artifact.sideinfo_path.stat().st_size
            ratio = (original_size / encoded_size) if encoded_size else 0.0
            digest = blake3(artifact.token_path.read_bytes()).hexdigest()

            start_dt = origin + timedelta(seconds=chunk.start_time)
            end_dt = origin + timedelta(seconds=chunk.end_time)

            manifest.append(
                ManifestRecord(
                    manifest_id=str(uuid4()),
                    tenant_id=config.project.default_tenant_id,
                    stream_id=f"{chunk.label}:{chunk.video_path.stem}",
                    label=chunk.label,
                    t0=start_dt.isoformat(),
                    t1=end_dt.isoformat(),
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    codebook_id=config.project.default_codebook_id,
                    model_id=config.project.default_model_id,
                    chunk_path=str(chunk.video_path),
                    token_uri=str(artifact.token_path),
                    sideinfo_uri=str(artifact.sideinfo_path),
                    embedding_path=str(embedding_path),
                    byte_size=encoded_size,
                    ratio=ratio,
                    hash=f"blake3:{digest}",
                    quality_stats=quality,
                    tags=[chunk.label],
                )
            )

        return manifest


def _compute_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Estimate PSNR and a SSIM-based proxy for VMAF across frames."""

    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    psnr_values = []
    ssim_values = []
    for reference, candidate in zip(original, reconstructed):
        psnr_values.append(peak_signal_noise_ratio(reference, candidate, data_range=255))
        ssim_values.append(
            structural_similarity(reference, candidate, data_range=255, channel_axis=-1)
        )
    return {
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "vmaf": float(np.mean(ssim_values) * 100) if ssim_values else 0.0,
    }
