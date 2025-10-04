"""FastAPI server exposing the conversational search API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from transformers import CLIPModel, CLIPTokenizer

from src.api.auth import verify_api_key
from src.api.models import FeedbackRequest, SearchRequest, SearchResponse, SearchResult
from src.api.rate_limit import enforce_rate_limit
from src.config import AppConfig, load_config
from src.core.generation import generate_answer
from src.core.retrieval import Retriever, expand_query
from src.utils import REQUEST_COUNTER, REQUEST_LATENCY, configure_logging
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class QueryEmbedder:
    """Encodes user queries into embeddings."""

    model: CLIPModel
    tokenizer: CLIPTokenizer
    device: str

    @classmethod
    def from_config(cls, config: AppConfig) -> "QueryEmbedder":
        model = CLIPModel.from_pretrained(config.models.clip_model_name)
        tokenizer = CLIPTokenizer.from_pretrained(config.models.clip_model_name)
        device = (
            torch.device("cuda")
            if config.models.device == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=str(device))

    def embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        with torch.no_grad():  # type: ignore[name-defined]
            features = self.model.get_text_features(**tokens)
        embedding = features.cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        return embedding.squeeze(0)


@lru_cache(maxsize=1)
def get_embedder(config: AppConfig) -> QueryEmbedder:
    return QueryEmbedder.from_config(config)


@dataclass
class ApplicationState:
    config: AppConfig
    retriever: Retriever
    embedder: QueryEmbedder


def build_app() -> FastAPI:
    config = load_config()
    configure_logging(config)
    LOGGER.info("Starting API server", extra={"env": config.project.environment})

    metadata_path = config.data.processed_dir / "manifest.json"
    if not metadata_path.exists():
        raise RuntimeError("Metadata file not found. Run run_pipeline.py first.")

    retriever = Retriever(config, metadata_path)
    embedder = get_embedder(config)

    state = ApplicationState(config=config, retriever=retriever, embedder=embedder)

    app = FastAPI(title="RAG Video Search", version="1.0.0")
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": state.config.project.environment}

    @router.post("/search", response_model=SearchResponse)
    async def search(payload: SearchRequest, api_key: str = Depends(verify_api_key)) -> SearchResponse:
        start = time.perf_counter()
        status_code = "200"
        try:
            await enforce_rate_limit(api_key)
            query_embedding = state.embedder.embed(payload.query)
            history_embeddings = [
                state.embedder.embed(item.content) for item in payload.history[-state.config.api.max_history :]
            ]
            if payload.options.expand and history_embeddings:
                query_embedding = expand_query(query_embedding, history_embeddings)
            results = state.retriever.search(query_embedding, top_k=payload.options.top_k)
            context = [f"{item.label} {item.start_time:.1f}-{item.end_time:.1f}s" for item in results]
            answer = generate_answer(
                payload.query,
                context,
                model_name=state.config.models.generator_model_name,
            )
            response = SearchResponse(
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
            REQUEST_LATENCY.labels(endpoint="/search", method="POST").observe(duration)
            REQUEST_COUNTER.labels(endpoint="/search", method="POST", status=status_code).inc()

    @router.post("/feedback")
    async def feedback(payload: FeedbackRequest, api_key: str = Depends(verify_api_key)) -> dict[str, str]:
        await enforce_rate_limit(api_key)
        LOGGER.info("Received feedback", extra={"api_key": api_key, "helpful": payload.helpful})
        return {"status": "accepted"}

    @router.get("/metrics")
    async def metrics() -> PlainTextResponse:
        data = generate_latest()
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)

    static_dir = state.config.frontend.static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app
