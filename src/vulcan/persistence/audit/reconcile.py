"""Audit transaction lifecycle reconciliation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TransactionLifecycle:
    transaction_id: str
    prepared: str | None = None
    terminal: str | None = None

def reconcile_transactions(events: tuple[object, ...]) -> dict[str, TransactionLifecycle]:
    from vulcan.persistence.audit.events import ABORT_EVENTS, COMMIT_EVENTS, PREPARED_EVENTS
    out: dict[str, TransactionLifecycle] = {}
    for e in events:
        t = getattr(e, "event_type")
        d = getattr(e, "data")
        tx = d.get("transaction_id") if isinstance(d, dict) else None
        if not isinstance(tx, str):
            continue
        current = out.get(tx, TransactionLifecycle(tx))
        if t in PREPARED_EVENTS:
            if current.prepared is not None:
                raise ValueError("duplicate transaction prepare")
            current = TransactionLifecycle(tx, t, current.terminal)
        if t in COMMIT_EVENTS or t in ABORT_EVENTS:
            if current.terminal is not None:
                raise ValueError("duplicate transaction terminal")
            current = TransactionLifecycle(tx, current.prepared, t)
        out[tx] = current
    for tx, lifecycle in out.items():
        if lifecycle.terminal and lifecycle.prepared is None:
            raise ValueError(f"transaction {tx} terminal without prepare")
    return out
