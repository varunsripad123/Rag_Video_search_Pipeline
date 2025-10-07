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
    # Auto-label filters (Zero-Copy AI)
    filter_objects: Optional[List[str]] = Field(None, description="Filter by detected objects (e.g., ['person', 'car'])")
    filter_action: Optional[str] = Field(None, description="Filter by action (e.g., 'walking')")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum auto-label confidence")


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
    auto_labels: Optional[dict] = Field(None, description="Auto-generated labels (objects, actions, captions)")


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


# Auto-labeling models (Zero-Copy AI)
class AutoLabelRequest(BaseModel):
    """Request to auto-label a video chunk."""
    manifest_id: str
    include_audio: bool = Field(True, description="Include audio transcription")


class AutoLabelResponse(BaseModel):
    """Response with auto-generated labels."""
    manifest_id: str
    objects: List[str] = Field(default_factory=list, description="Detected objects")
    object_counts: dict = Field(default_factory=dict, description="Object occurrence counts")
    action: str = Field("unknown", description="Recognized action")
    action_confidence: float = Field(0.0, ge=0.0, le=1.0)
    caption: str = Field("", description="Generated caption")
    audio_text: str = Field("", description="Transcribed audio")
    has_speech: bool = Field(False)
    audio_language: str = Field("unknown")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    metadata: dict = Field(default_factory=dict)
