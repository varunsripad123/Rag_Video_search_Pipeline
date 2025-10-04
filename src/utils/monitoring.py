"""Monitoring utilities using Prometheus client."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNTER = Counter(
    "rag_video_requests_total", "Number of API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "rag_video_request_latency_seconds", "Latency of API requests", ["endpoint", "method"]
)
PIPELINE_DURATION = Histogram(
    "rag_video_pipeline_duration_seconds", "Duration of pipeline stages", ["stage"]
)
INDEX_SIZE = Gauge("rag_video_index_size", "Number of vectors in the FAISS index")


@contextmanager
def track_stage(stage: str) -> Iterator[None]:
    """Context manager to record pipeline stage duration."""

    start = time.perf_counter()
    try:
        yield
    finally:
        PIPELINE_DURATION.labels(stage=stage).observe(time.perf_counter() - start)
