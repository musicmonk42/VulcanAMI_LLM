"""Integrity-bound append-time secondary indexes for canonical audit v2."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

INDEX_VERSION = "vulcan-audit-index/1"
INDEX_KEYS = ("episode","transaction","actor_digest","capability","policy","domain","memory_record","proposal","release","incident")
FIELD_BY_INDEX = {"episode":"case_id","transaction":"transaction_id","actor_digest":"actor_digest","capability":"capability","policy":"policy_id","domain":"domain","memory_record":"record_id","proposal":"proposal_digest","release":"release_id","incident":"incident_id"}

@dataclass(frozen=True, slots=True)
class AuditPage:
    events: tuple[object, ...]
    next_cursor: str | None

def empty_index() -> dict[str, dict[str, list[int]]]:
    return {k: {} for k in INDEX_KEYS}

def add_event(index: dict[str, dict[str, list[int]]], sequence: int, data: dict[str, object]) -> None:
    for name, field in FIELD_BY_INDEX.items():
        value = data.get(field)
        if isinstance(value, str):
            index[name].setdefault(value, []).append(sequence)

def index_digest(source_digest: str, index: dict[str, dict[str, list[int]]], canonical) -> str:
    import hashlib
    return hashlib.sha256(canonical({"source_digest": source_digest, "index": index}).encode() if isinstance(canonical({"source_digest": source_digest, "index": index}), str) else canonical({"source_digest": source_digest, "index": index})).hexdigest()

def write_index(path: Path, *, source_digest: str, index: dict[str, dict[str, list[int]]], canonical, sha) -> None:
    payload = {"schema_version": INDEX_VERSION, "source_digest": source_digest, "index": index}
    payload["index_digest"] = sha(canonical(payload))
    path.write_bytes(canonical(payload) + b"\n")

def read_index(path: Path, *, source_digest: str, canonical, sha, loads) -> dict[str, dict[str, list[int]]] | None:
    try:
        payload = loads(path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != INDEX_VERSION or payload.get("source_digest") != source_digest:
        return None
    digest = payload.get("index_digest"); check = dict(payload); check.pop("index_digest", None)
    if digest != sha(canonical(check)):
        return None
    idx = payload.get("index")
    if not isinstance(idx, dict) or set(idx) != set(INDEX_KEYS):
        return None
    return idx  # verified shape enough; source verification validates sequences

def paginate_sequences(sequences: Iterable[int], *, cursor: str | None, limit: int) -> tuple[tuple[int, ...], str | None]:
    if limit < 1 or limit > 1000:
        raise ValueError("page limit bound")
    start = 0 if cursor is None else int(cursor)
    if start < 0:
        raise ValueError("invalid cursor")
    items = tuple(sequences)
    page = items[start:start + limit]
    next_cursor = None if start + limit >= len(items) else str(start + limit)
    return page, next_cursor
