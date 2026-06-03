"""JSON normalization helpers for loosely typed agent payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
