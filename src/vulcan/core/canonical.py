"""Canonical JSON and digest helpers for bounded primitive values."""
from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence

from vulcan.core.time import format_utc

MAX_DEPTH = 24
MAX_ITEMS = 10_000
MAX_STRING = 16_384


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize(value: object, *, depth: int, count: list[int]) -> object:
    if depth > MAX_DEPTH:
        raise ValueError("canonical JSON maximum depth exceeded")
    count[0] += 1
    if count[0] > MAX_ITEMS:
        raise ValueError("canonical JSON maximum item count exceeded")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not canonical")
        return value
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        if len(normalized) > MAX_STRING:
            raise ValueError("string exceeds canonical length bound")
        if any(ord(ch) < 0x20 for ch in normalized):
            raise ValueError("control characters are not canonical")
        return normalized
    if isinstance(value, datetime):
        return format_utc(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize(asdict(value), depth=depth + 1, count=count)
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        seen_normalized: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            norm_key = _normalize(key, depth=depth + 1, count=count)
            if norm_key in seen_normalized:
                raise ValueError("canonical JSON key collision after normalization")
            seen_normalized.add(norm_key)
            out[norm_key] = _normalize(item, depth=depth + 1, count=count)
        return {key: out[key] for key in sorted(out)}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize(item, depth=depth + 1, count=count) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonicalize(value: object) -> object:
    return _normalize(value, depth=0, count=[0])


def canonical_json(value: object) -> bytes:
    normalized = canonicalize(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_digest(value: object) -> str:
    return sha256_bytes(canonical_json(value))
