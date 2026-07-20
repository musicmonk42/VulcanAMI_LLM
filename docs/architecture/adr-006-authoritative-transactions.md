# ADR-006: Shared authoritative transaction protocol

Status: Accepted (draft PR P16)

## Context

Audit, alignment, domain, memory, learning, CSIU, and improvement subsystems mutate different durable stores. A single database would create a false central authority, but ad hoc state names make recovery ambiguous.

## Decision

`vulcan.persistence.transactions` is the shared contract. Each subsystem keeps its durable authority and records the same immutable lifecycle:

1. `prepared` reserves an idempotency key and expected revision/CAS boundary.
2. `persisted` means the subsystem authority durably accepted the mutation.
3. `audit_committed` means the required audit authority durably recorded the event.
4. `published` is the only externally visible success point.
5. `aborted` is a terminal no-effect outcome.
6. `stale_cas` is a normal terminal outcome when the expected revision is stale and no effect ambiguity exists.
7. `manual_recovery` is reserved for ambiguous authoritative effects.

Every event carries transaction ID, actor/principal digest, target identity, prior revision/digest, proposed digest, result category, and canonical event digest. Startup reconciliation registers one reconciler per subsystem and asks it to resolve prepared or ambiguous records.

Injected authorities are borrowed unless explicitly marked owned. Closing borrowed handles must not cascade into injected databases, audit logs, providers, or resource authorities.

## Consequences

Later persistence subsystems implement a common state machine and recovery vocabulary without sharing a database. Stale CAS no longer pages manual recovery unless an external effect might have occurred but cannot be determined.

## Migration

Existing stores can adopt this protocol through one adapter per store pattern (for example file+audit or SQLite+audit). Rollback is to keep subsystem-local mutation code while preserving recorded transaction events as audit evidence.
