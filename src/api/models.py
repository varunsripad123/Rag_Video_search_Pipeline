"""Pydantic models for API requests and responses."""
from __future__ import annotations

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
