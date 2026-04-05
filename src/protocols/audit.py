"""Canonical audit protocol and implementation.

Lightweight JSONL backend with tamper-evident hash chain.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    event_type: str
    data: dict
    severity: str = "INFO"
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""
    event_hash: str = ""


@runtime_checkable
class AuditProtocol(Protocol):
    """Interface for audit logging implementations."""

    def log_event(
        self, event_type: str, data: dict, severity: str = "INFO"
    ) -> None: ...

    def get_events(
        self,
        *,
        event_type: str | None = None,
        agent_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict]: ...

    def verify_integrity(self) -> bool: ...

    def shutdown(self) -> None: ...


class AuditLogger:
    """Canonical audit logger with tamper-evident hash chain."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        max_memory_events: int = 1000,
        redact_patterns: list[str] | None = None,
    ):
        self._events: list[AuditEvent] = []
        self._lock = threading.RLock()
        self._last_hash = "GENESIS"
        self._max_memory = max_memory_events
        self._redact_patterns = redact_patterns or []
        self._log_file: Path | None = None
        if log_dir:
            self._log_file = Path(log_dir) / "audit.jsonl"
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, event: AuditEvent) -> str:
        payload = json.dumps(
            {"type": event.event_type, "data": event.data,
             "ts": event.timestamp, "prev": event.prev_hash},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _redact(self, data: dict) -> dict:
        if not self._redact_patterns:
            return data
        redacted = {}
        for k, v in data.items():
            if any(p in k.lower() for p in self._redact_patterns):
                redacted[k] = "[REDACTED]"
            elif isinstance(v, dict):
                redacted[k] = self._redact(v)
            else:
                redacted[k] = v
        return redacted

    def log_event(
        self, event_type: str, data: dict, severity: str = "INFO"
    ) -> None:
        with self._lock:
            event = AuditEvent(
                event_type=event_type,
                data=self._redact(data),
                severity=severity,
                prev_hash=self._last_hash,
            )
            event.event_hash = self._compute_hash(event)
            self._last_hash = event.event_hash
            self._events.append(event)
            if len(self._events) > self._max_memory:
                self._events = self._events[-self._max_memory:]
            if self._log_file:
                try:
                    with open(self._log_file, "a") as f:
                        f.write(json.dumps(asdict(event)) + "\n")
                except OSError:
                    logger.warning("Failed to write audit event to file")

    def get_events(
        self,
        *,
        event_type: str | None = None,
        agent_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        with self._lock:
            filtered = self._events
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            if agent_id:
                filtered = [
                    e for e in filtered
                    if e.data.get("agent_id") == agent_id
                ]
            if since:
                filtered = [e for e in filtered if e.timestamp >= since]
            return [asdict(e) for e in filtered[-limit:]]

    def verify_integrity(self) -> bool:
        with self._lock:
            prev = "GENESIS"
            for event in self._events:
                if event.prev_hash != prev:
                    return False
                expected = self._compute_hash(event)
                if event.event_hash != expected:
                    return False
                prev = event.event_hash
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_hash = "GENESIS"
