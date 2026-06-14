"""Structured logging decorator for execution-path functions."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any


def _summarize(obj: Any) -> str:
    """Convert a return value to a compact, log-safe string."""
    if obj is None:
        return "None"
    if isinstance(obj, bool):
        return str(obj)
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return f"str[{len(obj)}]"
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    if isinstance(obj, dict):
        return f"dict{{{len(obj)} keys}}"
    if isinstance(obj, tuple):
        return f"tuple[{len(obj)}]"
    # Custom types: show key attributes if available
    cls = type(obj).__name__
    attrs = []
    for key in ("id", "type", "state", "kind"):
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                attrs.append(f"{key}={val}")
    if attrs:
        return f"{cls}({', '.join(attrs)})"
    return cls


def logged(logger: logging.Logger):
    """Decorator factory: logs function entry, success exit, and failure.

    Usage::

        _log = logging.getLogger(__name__)

        @logged(_log)
        async def run(self, user_input: str) -> Task | None:
            ...
    """

    def _decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args, **kwargs):
                logger.info("%s: called", func.__qualname__)
                try:
                    result = await func(*args, **kwargs)
                    logger.info("%s: done, result=%s", func.__qualname__, _summarize(result))
                    return result
                except Exception:
                    logger.exception("%s: failed", func.__qualname__)
                    raise

            return _async_wrapper

        @functools.wraps(func)
        def _sync_wrapper(*args, **kwargs):
            logger.info("%s: called", func.__qualname__)
            try:
                result = func(*args, **kwargs)
                logger.info("%s: done, result=%s", func.__qualname__, _summarize(result))
                return result
            except Exception:
                logger.exception("%s: failed", func.__qualname__)
                raise

        return _sync_wrapper

    return _decorator
