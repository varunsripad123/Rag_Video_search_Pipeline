"""Utility exports."""

from importlib import import_module
from typing import Any

from .logging import configure_logging, get_logger
from .monitoring import INDEX_SIZE, PIPELINE_DURATION, REQUEST_COUNTER, REQUEST_LATENCY, track_stage

__all__ = [
    "video",
    "configure_logging",
    "get_logger",
    "INDEX_SIZE",
    "PIPELINE_DURATION",
    "REQUEST_COUNTER",
    "REQUEST_LATENCY",
    "track_stage",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - import hook
    if name == "video":
        return import_module("src.utils.video")
    raise AttributeError(f"module 'src.utils' has no attribute '{name}'")
