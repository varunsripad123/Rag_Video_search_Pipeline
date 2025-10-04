"""FastAPI application factory."""

from __future__ import annotations

from typing import Any

__all__ = ["build_app"]


def build_app(*args: Any, **kwargs: Any):
    from .server import build_app as _build_app

    return _build_app(*args, **kwargs)
