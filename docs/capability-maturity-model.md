# Capability maturity model

Capability reporting is a governance function owned by the cognitive microkernel. A constructed object, import success, route presence, test filename, or marketing text is not enough to advertise a capability.

## Statuses

`ABSENT`, `RESEARCH`, `SHADOW`, `EVALUATED`, `ADMITTED`, `ACTIVE`, `DEGRADED`, `SUSPENDED`, and `RETIRED` are the only valid statuses. Only `ACTIVE` and explicitly `DEGRADED` capabilities may appear in public production capability responses. Operator-scoped views may include research and shadow records with their limitations.

## Evidence requirements

Each record binds a capability ID, implementation digest, release digest, route/port reachability, owner, evaluation artifact, safety and impact artifacts, active policy digest, rollback method, limitations, review date, expiry timestamp, and dependencies. Evidence artifacts must exist, match full SHA-256 digests, use exact schemas, and remain unexpired.

## Current migrated records

The first real migrated public record is `cap.bounded_arithmetic`, limited to NFC-normalized `und` bounded arithmetic on the canonical chat route. Broad reasoning is research, internal LLM serving is suspended without an admitted model release, and learning/self-improvement remain shadow-only until governed promotion evidence exists.

## Dependency policy

An `ACTIVE` capability is transitively suspended when any dependency is missing, expired, suspended, or below `ADMITTED`. This prevents language realization from advertising production readiness when its verified model release or fidelity verifier is not admitted.
