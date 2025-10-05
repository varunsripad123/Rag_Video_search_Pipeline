"""Local pytest configuration and lightweight asyncio support."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``asyncio`` marker for compatibility."""
    config.addinivalue_line("markers", "asyncio: mark test to run with asyncio event loop")


def _run_coroutine(func: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(func(**kwargs))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute coroutine tests without requiring pytest-asyncio."""
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func) or pyfuncitem.get_closest_marker("asyncio"):
        _run_coroutine(test_func, pyfuncitem.funcargs)
        return True
    return None
