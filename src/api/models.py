"""Pydantic models for API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class SearchOptions(BaseModel):
    expand: bool = True
    top_k: int = Field(5, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str
    history: List[Message] = Field(default_factory=list)
    options: SearchOptions = SearchOptions()


class SearchResult(BaseModel):
    manifest_id: str
    label: str
    score: float
    start_time: float
    end_time: float
    asset_url: str


class SearchResponse(BaseModel):
    answer: str
    results: List[SearchResult]


class FeedbackRequest(BaseModel):
    query: str
    helpful: bool
    notes: Optional[str] = None


class IngestInitRequest(BaseModel):
    label: str
    stream_id: Optional[str] = None


class IngestInitResponse(BaseModel):
    stream_id: str
    upload_uri: str


class IngestChunkResponse(BaseModel):
    manifest_ids: List[str]
    ratio: float
    codebook_id: str


class TimeRange(BaseModel):
    start: datetime
    end: datetime


class SimilarSearchRequest(SearchRequest):
    stream_id: Optional[str] = None
    time_range: Optional[TimeRange] = None


class SimilarSearchResponse(SearchResponse):
    pass


class AnomalyAggregateRequest(BaseModel):
    stream_id: str
    time_range: TimeRange
    granularity_s: int = Field(60, ge=1)


class AnomalyPoint(BaseModel):
    timestamp: datetime
    score: float


class AnomalyAggregateResponse(BaseModel):
    points: List[AnomalyPoint]


class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DecodeRequest(BaseModel):
    manifest_id: str
    roi: Optional[ROI] = None
    fps: Optional[float] = None
    quality: str = Field("standard", pattern=r"standard|preview|analysis")


class RDPoint(BaseModel):
    bitrate: float
    psnr: float


class RDCurveResponse(BaseModel):
    points: List[RDPoint]
