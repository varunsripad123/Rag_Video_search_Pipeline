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
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from transformers import CLIPModel, CLIPTokenizer

from src.api.auth import verify_api_key
from src.api.models import (
    AnomalyAggregateRequest,
    AnomalyAggregateResponse,
    AnomalyPoint,
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
from src.core.codecs import NeuralVideoCodec
from src.core.embedding import EmbeddingExtractor
from src.core.generation import generate_answer
from src.core.pipeline import build_manifest_entry
from src.core.retrieval import Retriever, expand_query
from src.utils import REQUEST_COUNTER, REQUEST_LATENCY, configure_logging
from src.utils.io import ManifestEntry, read_manifest, write_manifest
from src.utils.logging import get_logger
from src.utils.video import VideoChunk, load_video_frames, probe_fps, write_video

LOGGER = get_logger(__name__)


@dataclass
class QueryEmbedder:
    """Encodes user queries into embeddings with CLIP fallback support."""

    model: Optional[CLIPModel]
    tokenizer: Optional[CLIPTokenizer]
    device: torch.device
    fallback_dim: int = 512

    @classmethod
    def from_config(cls, config: AppConfig) -> "QueryEmbedder":
        device = torch.device(
            "cuda" if config.models.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        try:
            model = CLIPModel.from_pretrained(config.models.clip_model_name).to(device)
            tokenizer = CLIPTokenizer.from_pretrained(config.models.clip_model_name)
            model.eval()
            LOGGER.info("Loaded CLIP text encoder", extra={"device": str(device)})
            return cls(model=model, tokenizer=tokenizer, device=device)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CLIP model unavailable, falling back to hashed text embeddings", exc_info=exc)
            return cls(model=None, tokenizer=None, device=device)

    def embed(self, text: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            return self._fallback_embed(text)
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**tokens)
        embedding = features.cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        return embedding.squeeze(0)

    def _fallback_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dim, dtype=np.float32)
        for idx, byte in enumerate(text.encode("utf-8")):
            vector[idx % self.fallback_dim] += (byte / 255.0)
        norm = np.linalg.norm(vector) + 1e-8
        return (vector / norm).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder(config: AppConfig) -> QueryEmbedder:
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
    extractor = EmbeddingExtractor(config)
    extractor.configure_precision()
    extractor.load()
    codec = NeuralVideoCodec(config.codec, device=config.models.device)

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
            query_embedding = state.embedder.embed(payload.query)
            history_embeddings = [
                state.embedder.embed(item.content) for item in payload.history[-state.config.api.max_history :]
            ]
            if payload.options.expand and history_embeddings:
                query_embedding = expand_query(query_embedding, history_embeddings)
            results = state.retriever.search(query_embedding, top_k=payload.options.top_k)
            context = [
                f"{item.label} {item.start_time:.1f}-{item.end_time:.1f}s" for item in results
            ]
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

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        data = generate_latest()
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)

    static_dir = state.config.frontend.static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

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
