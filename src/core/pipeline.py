import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from uuid import uuid4

import numpy as np
import torch

from src.config import AppConfig
from src.core.chunking import chunk_dataset, persist_metadata
from src.core.embedding import EmbeddingExtractor
from src.core.indexing import build_index
from src.core.codecs import NeuralVideoCodec, EncodedChunk
from src.utils import track_stage
from src.utils.io import ManifestEntry
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames

LOGGER = get_logger(__name__)


def sha256_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def _quality_metrics(original: np.ndarray, decoded: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics between original and decoded frames."""
    # Ensure shapes match
    if original.shape != decoded.shape:
        LOGGER.warning(f"Shape mismatch: original {original.shape} vs decoded {decoded.shape}")
        # Resize decoded to match original if needed
        min_t = min(original.shape[0], decoded.shape[0])
        original = original[:min_t]
        decoded = decoded[:min_t]
    
    # Compute MSE
    mse = np.mean((original.astype(float) - decoded.astype(float)) ** 2)
    
    # Compute PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
    }


def run_pipeline(config: AppConfig, enable_auto_labeling: bool = False) -> Path:
    """
    Run the complete video processing pipeline.
    
    Args:
        config: Application configuration
        enable_auto_labeling: If True, run auto-labeling (YOLO, BLIP-2, Whisper, VideoMAE)
    
    Returns:
        Path to the built FAISS index
    """
    chunks = list(chunk_dataset(config))
    embeddings_dir = config.data.processed_dir / "embeddings"
    tokens_dir = config.data.processed_dir / "tokens"
    sideinfo_dir = config.data.processed_dir / "sideinfo"
    for d in (embeddings_dir, tokens_dir, sideinfo_dir):
        d.mkdir(parents=True, exist_ok=True)

    extractor = EmbeddingExtractor(config)
    extractor.configure_precision()
    extractor.load()
    codec = NeuralVideoCodec(config.codec, device=config.models.device)
    
    # Initialize auto-labeler if enabled
    auto_labeler = None
    if enable_auto_labeling:
        try:
            from src.core.labeling import AutoLabeler
            LOGGER.info("Initializing auto-labeling models...")
            auto_labeler = AutoLabeler(config)
            auto_labeler.load()
            LOGGER.info("Auto-labeling enabled")
        except Exception as e:
            LOGGER.warning(f"Failed to initialize auto-labeler: {e}")
            auto_labeler = None

    manifest: List[ManifestEntry] = []
    processed_count = 0
    for chunk in chunks:
        try:
            processed_count += 1
            LOGGER.info(f"Processing chunk {processed_count}/{len(chunks)}: {chunk.video_path.name}")
            
            # Load video frames
            try:
                tensor = load_video_frames(chunk.video_path)  # (1,3,T,H,W)
                LOGGER.info(f"Loaded frames with shape: {tensor.shape}")
            except Exception as e:
                LOGGER.error(f"Failed to load video frames: {e}", exc_info=True)
                continue
            
            # Encode to tokens
            try:
                enc_chunk: EncodedChunk = codec.encode_tensor(tensor, tokens_dir / f"{chunk.video_path.stem}.npz")
                LOGGER.info(f"Encoded to tokens: {enc_chunk.size_bytes} bytes")
            except Exception as e:
                LOGGER.error(f"Failed to encode to tokens: {e}", exc_info=True)
                continue
            
            # Quality metrics
            try:
                decoded = codec.decode(enc_chunk.path)
                original = (tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)  # back to (T,H,W,3)
                quality = _quality_metrics(original, decoded)
            except Exception as e:
                LOGGER.warning(f"Failed to compute quality metrics: {e}")
                quality = {"mse": 0.0, "psnr": 0.0}
            
            # Sideinfo
            try:
                sideinfo = {
                    "codebook_id": config.codec.codebook_id,
                    "layout": "tiles16x16",
                    "dims": original.shape[1:3],
                    "time_stride": original.shape[0],
                }
                (sideinfo_dir / f"{chunk.video_path.stem}.json").write_text(
                    json.dumps(sideinfo, indent=2), encoding="utf-8"
                )
            except Exception as e:
                LOGGER.warning(f"Failed to write sideinfo: {e}")
            
            # Embedding - pass the original tensor
            try:
                LOGGER.info(f"Extracting embeddings...")
                embedding = extractor.encode_frames(tensor)
                LOGGER.info(f"Embedding shape: {embedding.shape}")
                np.save(embeddings_dir / f"{chunk.video_path.stem}.npy", embedding)
            except Exception as e:
                LOGGER.error(f"Failed to extract embeddings: {e}", exc_info=True)
                continue
            
            # Auto-labeling (optional)
            auto_labels = None
            if auto_labeler is not None:
                try:
                    LOGGER.info(f"Running auto-labeling...")
                    # Convert tensor to numpy frames for labeling
                    frames_np = tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                    frames_list = [(frames_np[i] * 255).astype(np.uint8) for i in range(frames_np.shape[0])]
                    
                    auto_labels = auto_labeler.label_video_chunk(
                        chunk.video_path,
                        frames=frames_list,
                        include_audio=True
                    )
                    LOGGER.info(
                        f"Auto-labels: objects={auto_labels.get('objects', [])[:3]}, "
                        f"action={auto_labels.get('action')}, "
                        f"caption='{auto_labels.get('caption', '')[:50]}...'"
                    )
                except Exception as e:
                    LOGGER.warning(f"Auto-labeling failed: {e}")
                    auto_labels = None
            
            # Manifest entry
            entry = ManifestEntry(
                manifest_id=str(uuid4()),
                tenant_id=config.data.default_tenant,
                stream_id=chunk.video_path.stem.split("_chunk")[0],
                label=chunk.label,
                chunk_path=str(chunk.video_path),
                token_path=str(enc_chunk.path),
                sideinfo_path=str(sideinfo_dir / f"{chunk.video_path.stem}.json"),
                embedding_path=str(embeddings_dir / f"{chunk.video_path.stem}.npy"),
                start_time=chunk.start_frame / chunk.fps,
                end_time=chunk.end_frame / chunk.fps,
                fps=chunk.fps,
                codebook_id=config.codec.codebook_id,
                model_id=config.codec.model_id,
                byte_size=enc_chunk.size_bytes,
                ratio=(chunk.video_path.stat().st_size / enc_chunk.size_bytes),
                hash=f"sha256:{sha256_hash(enc_chunk.path)}",
                tags=[chunk.label],
                quality_stats=quality,
                auto_labels=auto_labels  # Add auto-labeling results
            )
            manifest.append(entry)
            LOGGER.info(f"Successfully processed chunk {chunk.video_path.name}")
        except Exception as e:
            LOGGER.error(f"Failed processing {chunk.video_path}: {e}", exc_info=True)

    if not manifest:
        raise ValueError("Pipeline failed: no valid embeddings")

    persist_metadata(config.data.processed_dir / "metadata.json", manifest)
    return build_index(config, manifest)


def build_manifest_entry(
    config: AppConfig,
    codec: NeuralVideoCodec,
    extractor: EmbeddingExtractor,
    chunk: VideoChunk,
    frames: torch.Tensor,
    embeddings_dir: Path,
    tokens_dir: Path,
    sideinfo_dir: Path,
    auto_labeler: Optional[any] = None
) -> tuple[ManifestEntry, np.ndarray]:
    """Build a manifest entry for a single video chunk (used by API ingestion).
    
    Args:
        config: Application configuration
        codec: Neural video codec
        extractor: Embedding extractor
        chunk: Video chunk metadata
        frames: Video frames as tensor
        embeddings_dir: Directory to save embeddings
        tokens_dir: Directory to save tokens
        sideinfo_dir: Directory to save sideinfo
        auto_labeler: Optional auto-labeler instance
    
    Returns:
        Tuple of (ManifestEntry, embedding array)
    """
    
    # Encode to tokens
    enc_chunk: EncodedChunk = codec.encode_tensor(frames, tokens_dir / f"{chunk.video_path.stem}.npz")
    
    # Quality metrics
    try:
        decoded = codec.decode(enc_chunk.path)
        original = (frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
        quality = _quality_metrics(original, decoded)
    except Exception as e:
        LOGGER.warning(f"Failed to compute quality metrics: {e}")
        quality = {"mse": 0.0, "psnr": 0.0}
    
    # Sideinfo
    sideinfo = {
        "codebook_id": config.codec.codebook_id,
        "layout": "tiles16x16",
        "dims": original.shape[1:3],
        "time_stride": original.shape[0],
    }
    sideinfo_path = sideinfo_dir / f"{chunk.video_path.stem}.json"
    sideinfo_path.write_text(json.dumps(sideinfo, indent=2), encoding="utf-8")
    
    # Embedding
    embedding = extractor.encode_frames(frames)
    embedding_path = embeddings_dir / f"{chunk.video_path.stem}.npy"
    np.save(embedding_path, embedding)
    
    # Auto-labeling (optional)
    auto_labels = None
    if auto_labeler is not None:
        try:
            # Convert tensor to numpy frames for labeling
            frames_np = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            frames_list = [(frames_np[i] * 255).astype(np.uint8) for i in range(frames_np.shape[0])]
            
            auto_labels = auto_labeler.label_video_chunk(
                chunk.video_path,
                frames=frames_list,
                include_audio=True
            )
            LOGGER.info(f"Auto-labeled: {auto_labels.get('objects', [])[:3]}")
        except Exception as e:
            LOGGER.warning(f"Auto-labeling failed: {e}")
            auto_labels = None
    
    # Manifest entry
    entry = ManifestEntry(
        manifest_id=str(uuid4()),
        tenant_id=config.data.default_tenant,
        stream_id=chunk.video_path.stem.split("_chunk")[0],
        label=chunk.label,
        chunk_path=str(chunk.video_path),
        token_path=str(enc_chunk.path),
        sideinfo_path=str(sideinfo_path),
        embedding_path=str(embedding_path),
        start_time=chunk.start_frame / chunk.fps,
        end_time=chunk.end_frame / chunk.fps,
        fps=chunk.fps,
        codebook_id=config.codec.codebook_id,
        model_id=config.codec.model_id,
        byte_size=enc_chunk.size_bytes,
        ratio=(chunk.video_path.stat().st_size / enc_chunk.size_bytes) if chunk.video_path.exists() else 0.0,
        hash=f"sha256:{sha256_hash(enc_chunk.path)}",
        tags=[chunk.label],
        quality_stats=quality,
        auto_labels=auto_labels  # Add auto-labeling results
    )
    
    return entry, embedding
