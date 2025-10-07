"""FastAPI server exposing the production video search API."""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Defer transformers import to avoid circular import issues
CLIPModel = None
CLIPTokenizer = None

def _try_import_clip():
    """Try to import CLIP at runtime to avoid circular imports."""
    global CLIPModel, CLIPTokenizer
    if CLIPModel is not None:
        return True
    try:
        # Try direct import from transformers.models.clip
        from transformers.models.clip import CLIPModel as _CLIPModel, CLIPTokenizer as _CLIPTokenizer
        CLIPModel = _CLIPModel
        CLIPTokenizer = _CLIPTokenizer
        return True
    except Exception as e1:
        try:
            # Fallback: Try standard import
            from transformers import CLIPModel as _CLIPModel, CLIPTokenizer as _CLIPTokenizer
            CLIPModel = _CLIPModel
            CLIPTokenizer = _CLIPTokenizer
            return True
        except Exception as e2:
            LOGGER.debug(f"Could not import CLIP: {e1}, {e2}")
            return False

from src.api.auth import verify_api_key
from src.api.models import (
    AnomalyAggregateRequest,
    AnomalyAggregateResponse,
    AnomalyPoint,
    AutoLabelRequest,
    AutoLabelResponse,
    DecodeRequest,
    FeedbackRequest,
    IngestChunkResponse,
    IngestInitRequest,
    IngestInitResponse,
    RDCurveResponse,
    RDPoint,
    ROI,
    SimilarSearchRequest,
    SimilarSearchResponse,
    SearchResult,
)
from src.api.rate_limit import enforce_rate_limit
from src.config import AppConfig, load_config
from src.core.retrieval import Retriever, expand_query
from src.utils import REQUEST_COUNTER, REQUEST_LATENCY, configure_logging
from src.utils.io import ManifestEntry, read_manifest, write_manifest
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames, probe_fps, write_video

# Defer these imports to avoid loading heavy models at module level
NeuralVideoCodec = None
EmbeddingExtractor = None
build_manifest_entry = None
generate_answer = None

def _try_import_pipeline_components():
    """Lazy import of pipeline components (only needed for ingestion)."""
    global NeuralVideoCodec, EmbeddingExtractor, build_manifest_entry, generate_answer
    if NeuralVideoCodec is not None:
        return True
    try:
        from src.core.codecs import NeuralVideoCodec as _NeuralVideoCodec
        from src.core.embedding import EmbeddingExtractor as _EmbeddingExtractor
        from src.core.pipeline import build_manifest_entry as _build_manifest_entry
        from src.core.generation import generate_answer as _generate_answer
        NeuralVideoCodec = _NeuralVideoCodec
        EmbeddingExtractor = _EmbeddingExtractor
        build_manifest_entry = _build_manifest_entry
        generate_answer = _generate_answer
        return True
    except Exception as e:
        LOGGER.error(f"Failed to import pipeline components: {e}")
        return False

def _fallback_generate_answer(query: str, context: list) -> str:
    """Simple answer generation fallback when LLM is not available."""
    if not context:
        return "No matching videos found."
    
    # Simple template-based answer
    n_results = len(context)
    if n_results == 1:
        return f"Found 1 relevant video: {context[0]}"
    else:
        top_results = context[:3]
        answer = f"Found {n_results} relevant videos. Top matches:\n"
        for i, ctx in enumerate(top_results, 1):
            answer += f"{i}. {ctx}\n"
        return answer.strip()

LOGGER = get_logger(__name__)


@dataclass
class QueryEmbedder:
    """Encodes user queries into embeddings with CLIP fallback support."""

    model: Optional[CLIPModel]
    tokenizer: Optional[CLIPTokenizer]
    device: torch.device
    target_dim: int = 512  # Match the index dimension (512 CLIP)
    fallback_dim: int = 512

    @classmethod
    def from_config(cls, config: AppConfig, target_dim: int = 512) -> "QueryEmbedder":
        device = torch.device(
            "cuda" if config.models.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        
        # Try to load CLIP (with better error handling)
        try:
            if _try_import_clip():
                import os
                
                # Get HF token from environment
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                
                LOGGER.info("Attempting to load CLIP model...")
                model = CLIPModel.from_pretrained(
                    config.models.clip_model_name,
                    token=token
                )
                model = model.to(device)
                model.eval()
                tokenizer = CLIPTokenizer.from_pretrained(
                    config.models.clip_model_name,
                    token=token
                )
                LOGGER.info("✅ CLIP model loaded successfully!")
                return cls(model=model, tokenizer=tokenizer, device=device, target_dim=target_dim)
        except Exception as e:
            LOGGER.warning(f"Failed to load CLIP: {e}")
        
        # Fallback: Use basic embeddings
        LOGGER.warning("⚠️  Using fallback text embeddings (reduced search quality)")
        LOGGER.warning("   To improve: pip install --upgrade transformers")
        return cls(model=None, tokenizer=None, device=device, target_dim=target_dim)

    def embed(self, text: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            embedding = self._fallback_embed(text)
        else:
            tokens = self.tokenizer(text, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**tokens)
            embedding = features.cpu().numpy()
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
            embedding = embedding.squeeze(0)
        
        # Pad to target dimension to match the index
        if embedding.shape[0] < self.target_dim:
            padding = np.zeros(self.target_dim - embedding.shape[0], dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        elif embedding.shape[0] > self.target_dim:
            embedding = embedding[:self.target_dim]
        
        return embedding

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Better fallback: Use word hashing with TF-IDF-like weighting."""
        import hashlib
        
        vector = np.zeros(self.fallback_dim, dtype=np.float32)
        
        # Tokenize and process words
        words = text.lower().split()
        if not words:
            words = ['empty']
        
        # Use word hashing with position weighting (earlier words matter more)
        for pos, word in enumerate(words):
            # Hash word to multiple positions for better distribution
            for i in range(3):  # Use 3 hash functions
                hash_val = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest(), 16)
                idx = hash_val % self.fallback_dim
                # Weight: earlier words get higher weight, use IDF-like decay
                weight = 1.0 / (1.0 + pos * 0.1)
                vector[idx] += weight
        
        # Add bigrams for better context
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            hash_val = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            idx = hash_val % self.fallback_dim
            vector[idx] += 0.5  # Bigrams get moderate weight
        
        # Normalize
        norm = np.linalg.norm(vector) + 1e-8
        return (vector / norm).astype(np.float32)


def get_embedder(config: AppConfig) -> QueryEmbedder:
    """Create a query embedder from config (no caching due to unhashable config)."""
    return QueryEmbedder.from_config(config)


@dataclass
class ApplicationState:
    """Container holding mutable server state."""

    config: AppConfig
    embedder: QueryEmbedder
    extractor: EmbeddingExtractor
    codec: NeuralVideoCodec
    retriever: Optional[Retriever]
    manifest_path: Path
    manifest: List[ManifestEntry] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    streams: Dict[str, Dict[str, str]] = field(default_factory=dict)
    auto_labeler: Optional[any] = field(default=None)  # Lazy-loaded on demand

    @property
    def embeddings_dir(self) -> Path:
        return self.config.data.processed_dir / "embeddings"

    @property
    def tokens_dir(self) -> Path:
        return self.config.data.processed_dir / "tokens"

    @property
    def sideinfo_dir(self) -> Path:
        return self.config.data.processed_dir / "sideinfo"

    def ensure_directories(self) -> None:
        for directory in (self.embeddings_dir, self.tokens_dir, self.sideinfo_dir):
            directory.mkdir(parents=True, exist_ok=True)


def build_app() -> FastAPI:
    """Factory returning the FastAPI application."""

    config = load_config()
    configure_logging(config)
    LOGGER.info("Starting API server", extra={"environment": config.project.environment})

    embedder = get_embedder(config)
    
    # Don't load extractor and codec for search-only mode (prevents CLIP loading issues)
    # These are only needed for video ingestion, not for search
    extractor = None
    codec = None
    LOGGER.info("Running in search-only mode (ingestion disabled)")

    manifest_path = config.data.processed_dir / "metadata.json"
    manifest: List[ManifestEntry] = []
    retriever: Optional[Retriever] = None
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        index_path = config.index.faiss_index_path
        if manifest and index_path.exists():
            retriever = Retriever(config, manifest_path)

    state = ApplicationState(
        config=config,
        embedder=embedder,
        extractor=extractor,
        codec=codec,
        retriever=retriever,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    state.ensure_directories()

    app = FastAPI(title="RAG Video Search", version="1.1.0")
    router = APIRouter(prefix="/v1")

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": state.config.project.environment}

    @router.post("/ingest/init", response_model=IngestInitResponse)
    async def ingest_init(
        payload: IngestInitRequest, api_key: str = Depends(verify_api_key)
    ) -> IngestInitResponse:
        await enforce_rate_limit(api_key)
        stream_id = payload.stream_id or f"{state.config.data.stream_prefix}_{uuid4().hex[:8]}"
        upload_dir = state.config.data.root_dir / payload.label
        upload_dir.mkdir(parents=True, exist_ok=True)
        state.streams[stream_id] = {"label": payload.label, "dir": str(upload_dir)}
        upload_uri = upload_dir / f"{stream_id}.mp4"
        return IngestInitResponse(stream_id=stream_id, upload_uri=f"file://{upload_uri.resolve()}")

    @router.post("/ingest/chunk", response_model=IngestChunkResponse)
    async def ingest_chunk(
        stream_id: str,
        t0: str,
        t1: str,
        file: UploadFile = File(...),
        api_key: str = Depends(verify_api_key),
    ) -> IngestChunkResponse:
        await enforce_rate_limit(api_key)
        if state.extractor is None or state.codec is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ingestion disabled (search-only mode)")
        if stream_id not in state.streams:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown stream_id")
        meta = state.streams[stream_id]
        label = meta["label"]
        upload_dir = Path(meta["dir"])
        async with state.lock:
            data = await file.read()
            chunk_name = f"{stream_id}_chunk_{uuid4().hex[:6]}.mp4"
            chunk_path = upload_dir / chunk_name
            chunk_path.write_bytes(data)

            frames = load_video_frames(chunk_path)
            fps = probe_fps(chunk_path, default=state.config.data.frame_rate)
            video_chunk = VideoChunk(
                video_path=chunk_path,
                label=label,
                start_frame=0,
                end_frame=frames.shape[0],
                fps=fps,
            )
            entry, embedding = build_manifest_entry(
                config=state.config,
                codec=state.codec,
                extractor=state.extractor,
                chunk=video_chunk,
                frames=frames,
                embeddings_dir=state.embeddings_dir,
                tokens_dir=state.tokens_dir,
                sideinfo_dir=state.sideinfo_dir,
            )
            start_ts = _parse_timestamp(t0)
            end_ts = _parse_timestamp(t1)
            entry.start_time = start_ts
            entry.end_time = end_ts

            state.manifest.append(entry)
            write_manifest(state.manifest_path, state.manifest)

            if state.retriever is None:
                from src.core.indexing import build_index

                build_index(state.config, state.manifest)
                state.retriever = Retriever(state.config, state.manifest_path)
            else:
                state.retriever.add_entry(entry, embedding)

        return IngestChunkResponse(
            manifest_ids=[entry.manifest_id],
            ratio=entry.ratio,
            codebook_id=entry.codebook_id,
        )

    @router.post("/search/similar", response_model=SimilarSearchResponse)
    async def search_similar(
        payload: SimilarSearchRequest, api_key: str = Depends(verify_api_key)
    ) -> SimilarSearchResponse:
        start = time.perf_counter()
        status_code = "200"
        await enforce_rate_limit(api_key)
        try:
            if state.retriever is None:
                return SimilarSearchResponse(answer="No indexed media yet.", results=[])
            
            # Use query expansion for better accuracy
            from .query_expansion import get_expanded_embedding
            query_embedding = get_expanded_embedding(payload.query, state.embedder, average=True)
            
            history_embeddings = [
                state.embedder.embed(item.content) for item in payload.history[-state.config.api.max_history :]
            ]
            if payload.options.expand and history_embeddings:
                query_embedding = expand_query(query_embedding, history_embeddings)
            
            # Search with auto-label filters
            results = state.retriever.search(
                query_embedding, 
                top_k=payload.options.top_k,
                filter_objects=payload.options.filter_objects,
                filter_action=payload.options.filter_action,
                min_confidence=payload.options.min_confidence
            )
            
            # Build context including auto-labels if available
            context = []
            for item in results:
                ctx = f"{item.label} {item.start_time:.1f}-{item.end_time:.1f}s"
                if item.auto_labels:
                    # Add caption and objects to context for better answers
                    caption = item.auto_labels.get('caption', '')
                    objects = ', '.join(item.auto_labels.get('objects', [])[:3])
                    if caption:
                        ctx += f" - {caption}"
                    elif objects:
                        ctx += f" - contains: {objects}"
                context.append(ctx)
            
            # Use LLM answer generation if available, otherwise use simple fallback
            if generate_answer is None:
                answer = _fallback_generate_answer(payload.query, context)
            else:
                answer = generate_answer(payload.query, context)
            response = SimilarSearchResponse(
                answer=answer,
                results=[
                    SearchResult(
                        manifest_id=item.manifest_id,
                        label=item.label,
                        score=item.score,
                        start_time=item.start_time,
                        end_time=item.end_time,
                        asset_url=item.asset_url,
                        auto_labels=item.auto_labels  # Include auto-labels in response
                    )
                    for item in results
                ],
            )
            return response
        except Exception as exc:  # noqa: BLE001
            status_code = "500"
            LOGGER.exception("Search failed", exc_info=exc)
            raise
        finally:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(endpoint="/v1/search/similar", method="POST").observe(duration)
            REQUEST_COUNTER.labels(endpoint="/v1/search/similar", method="POST", status=status_code).inc()

    @router.post("/aggregate/anomaly", response_model=AnomalyAggregateResponse)
    async def aggregate_anomaly(
        payload: AnomalyAggregateRequest, api_key: str = Depends(verify_api_key)
    ) -> AnomalyAggregateResponse:
        await enforce_rate_limit(api_key)
        start = payload.time_range.start
        end = payload.time_range.end
        window = timedelta(seconds=payload.granularity_s)
        points: List[AnomalyPoint] = []
        relevant = [
            entry
            for entry in state.manifest
            if entry.stream_id == payload.stream_id
            and start <= _to_datetime(entry.start_time, start) <= end
        ]
        if not relevant:
            return AnomalyAggregateResponse(points=[])
        embeddings = [np.load(entry.embedding_path) for entry in relevant]
        matrix = np.stack(embeddings)
        centroid = matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(matrix - centroid, axis=1)
        if distances.max() > 0:
            distances = distances / distances.max()
        for entry, score in zip(relevant, distances, strict=False):
            timestamp = _to_datetime(entry.start_time, start)
            bucket = start + ((timestamp - start) // window) * window
            points.append(AnomalyPoint(timestamp=bucket, score=float(score)))
        points.sort(key=lambda item: item.timestamp)
        return AnomalyAggregateResponse(points=points)

    @router.post("/decode")
    async def decode(payload: DecodeRequest, api_key: str = Depends(verify_api_key)) -> StreamingResponse:
        await enforce_rate_limit(api_key)
        if state.codec is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Decode disabled (search-only mode)")
        entry = _find_manifest(state, payload.manifest_id)
        frames = state.codec.decode(Path(entry.token_path))
        if payload.roi is not None:
            frames = _apply_roi(frames, payload.roi)
        if payload.quality == "preview":
            frames = frames[:, ::2, ::2, :]
        elif payload.quality == "analysis":
            frames = frames
        fps = payload.fps or entry.fps
        with io.BytesIO() as buffer:
            tmp_path = state.tokens_dir / f"preview_{uuid4().hex}.mp4"
            write_video(tmp_path, frames, fps)
            buffer.write(tmp_path.read_bytes())
            tmp_path.unlink(missing_ok=True)
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="video/mp4")

    @router.get("/stats/rd_curve", response_model=RDCurveResponse)
    async def rd_curve(stream_id: Optional[str] = None) -> RDCurveResponse:
        entries = [
            entry
            for entry in state.manifest
            if stream_id is None or entry.stream_id == stream_id
        ]
        if not entries:
            return RDCurveResponse(points=[])
        points = []
        for entry in entries:
            duration = max(entry.end_time - entry.start_time, 1e-6)
            bitrate = (entry.byte_size * 8) / duration
            points.append(RDPoint(bitrate=bitrate, psnr=entry.quality_stats.get("psnr", 0.0)))
        points.sort(key=lambda item: item.bitrate)
        return RDCurveResponse(points=points)

    @router.post("/feedback")
    async def feedback(payload: FeedbackRequest, api_key: str = Depends(verify_api_key)) -> dict[str, str]:
        await enforce_rate_limit(api_key)
        LOGGER.info("Received feedback", extra={"helpful": payload.helpful})
        return {"status": "accepted"}
    
    @router.post("/label/auto", response_model=AutoLabelResponse)
    async def auto_label_video(
        payload: AutoLabelRequest, 
        api_key: str = Depends(verify_api_key)
    ) -> AutoLabelResponse:
        """
        Generate automatic labels for a video chunk using AI models.
        
        Uses YOLO for object detection, VideoMAE for action recognition,
        BLIP-2 for caption generation, and Whisper for audio transcription.
        """
        await enforce_rate_limit(api_key)
        
        # Find the manifest entry
        entry = _find_manifest(state, payload.manifest_id)
        chunk_path = Path(entry.chunk_path)
        
        if not chunk_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Video file not found"
            )
        
        # Initialize auto-labeler if not already done
        if not hasattr(state, 'auto_labeler') or state.auto_labeler is None:
            try:
                from src.core.labeling import AutoLabeler
                LOGGER.info("Initializing auto-labeler for on-demand labeling")
                state.auto_labeler = AutoLabeler(state.config)
                state.auto_labeler.load()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Auto-labeling service unavailable: {str(e)}"
                )
        
        # Run auto-labeling
        try:
            labels = state.auto_labeler.label_video_chunk(
                chunk_path,
                frames=None,  # Will load frames automatically
                include_audio=payload.include_audio
            )
            
            # Update manifest entry with new labels
            async with state.lock:
                entry.auto_labels = labels
                write_manifest(state.manifest_path, state.manifest)
            
            # Return response
            return AutoLabelResponse(
                manifest_id=payload.manifest_id,
                objects=labels.get('objects', []),
                object_counts=labels.get('object_counts', {}),
                action=labels.get('action', 'unknown'),
                action_confidence=labels.get('action_confidence', 0.0),
                caption=labels.get('caption', ''),
                audio_text=labels.get('audio_text', ''),
                has_speech=labels.get('has_speech', False),
                audio_language=labels.get('audio_language', 'unknown'),
                confidence=labels.get('confidence', 0.0),
                metadata=labels.get('metadata', {})
            )
        
        except Exception as e:
            LOGGER.error(f"Auto-labeling failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Auto-labeling failed: {str(e)}"
            )
    
    @router.get("/video/{manifest_id}")
    async def get_original_video(
        manifest_id: str,
        api_key: str = Query(...),
        format: str = Query("mp4", regex="^(mp4|gif|thumbnail)$"),
        quality: str = Query("medium", regex="^(high|medium|low)$"),
        context: float = Query(0.0, ge=0, le=30)
    ) -> StreamingResponse:
        """
        Retrieve video segment with flexible options.
        
        Args:
            manifest_id: Unique segment identifier
            api_key: Authentication key
            format: Output format (mp4, gif, thumbnail)
            quality: Video quality (high, medium, low)
            context: Additional seconds before/after segment (0-30)
        """
        # Verify API key
        if api_key not in state.config.security.api_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        
        entry = _find_manifest(state, manifest_id)
        chunk_path = Path(entry.chunk_path)
        
        if not chunk_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video file not found: {chunk_path}")
        
        # If no special processing needed, return original
        if format == "mp4" and quality == "medium" and context == 0:
            from fastapi.responses import FileResponse
            return FileResponse(
                chunk_path,
                media_type="video/mp4",
                filename=chunk_path.name,
                headers={
                    "Content-Disposition": f'inline; filename="{chunk_path.name}"',
                    "Accept-Ranges": "bytes"
                }
            )
        
        # Extract custom segment
        from .segment_retrieval import SegmentRetriever
        
        start_time = entry.start_time
        end_time = entry.end_time
        
        # Add context if requested
        if context > 0:
            start_time, end_time = SegmentRetriever.get_segment_context(
                chunk_path, start_time, end_time, context
            )
        
        # Extract segment with requested format/quality
        output_path = SegmentRetriever.extract_segment(
            chunk_path, start_time, end_time, format, quality
        )
        
        # Determine media type
        media_types = {
            "mp4": "video/mp4",
            "gif": "image/gif",
            "thumbnail": "image/jpeg"
        }
        
        from fastapi.responses import FileResponse
        return FileResponse(
            output_path,
            media_type=media_types[format],
            filename=f"{manifest_id}.{format}",
            headers={
                "Content-Disposition": f'inline; filename="{manifest_id}.{format}"'
            }
        )

    @app.get("/demo")
    async def demo():
        """Serve the full demo interface directly."""
        from fastapi.responses import HTMLResponse, FileResponse
        try:
            static_base = Path(__file__).parent.parent.parent / "web" / "static"
            index_path = static_base / "index.html"
            if index_path.exists():
                return FileResponse(index_path, media_type="text/html")
        except Exception as e:
            LOGGER.error(f"Failed to load demo page: {e}")
        return HTMLResponse(content="<h1>Demo page not found</h1>")
    
    @app.get("/styles.css")
    async def styles():
        from fastapi.responses import FileResponse
        static_base = Path(__file__).parent.parent.parent / "web" / "static"
        return FileResponse(static_base / "styles.css", media_type="text/css")
    
    @app.get("/app.js")
    async def appjs():
        from fastapi.responses import FileResponse
        static_base = Path(__file__).parent.parent.parent / "web" / "static"
        return FileResponse(static_base / "app.js", media_type="application/javascript")
    
    @app.get("/")
    async def root():
        """Redirect to professional UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index_pro.html")
    
    @app.get("/demo")
    async def demo_old():
        """Old demo - redirect to new UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index_pro.html")

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        data = generate_latest()
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)

    static_dir = state.config.frontend.static_dir
    LOGGER.info(f"Static directory path: {static_dir}")
    LOGGER.info(f"Static directory exists: {static_dir.exists()}")
    
    if static_dir.exists():
        try:
            app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
            LOGGER.info(f"Mounted static files at /static from {static_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to mount static files: {e}")
    else:
        LOGGER.warning(f"Static directory does not exist: {static_dir}")

    return app


def _parse_timestamp(value: str) -> float:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp") from exc
    return dt.timestamp()


def _to_datetime(value: float, base: datetime) -> datetime:
    try:
        # If value already encodes an epoch second, reuse it.
        if value > 1e6:
            return datetime.fromtimestamp(value, tz=base.tzinfo)
        return base + timedelta(seconds=value)
    except Exception:  # noqa: BLE001
        return base


def _find_manifest(state: ApplicationState, manifest_id: str) -> ManifestEntry:
    for entry in state.manifest:
        if entry.manifest_id == manifest_id:
            return entry
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manifest not found")


def _apply_roi(frames: np.ndarray, roi: ROI) -> np.ndarray:
    y1 = max(0, roi.y)
    y2 = min(frames.shape[1], roi.y + roi.h)
    x1 = max(0, roi.x)
    x2 = min(frames.shape[2], roi.x + roi.w)
    if x1 >= x2 or y1 >= y2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ROI bounds")
    return frames[:, y1:y2, x1:x2, :]


__all__ = ["build_app"]
