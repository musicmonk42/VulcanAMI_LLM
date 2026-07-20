# ADR 005: Cognitive authority lattice

## Status

Accepted. This ADR defines the only authority escalation path for AMI cognition.

## Decision

Authority is a monotonic lattice:

1. `UNTRUSTED_PROPOSAL`: raw model, provider, user, tool, CSIU, learning, Graphix, or workspace output. It cannot mutate memory, execute effects, advertise capabilities, or become policy.
2. `VALIDATED_CANDIDATE`: schema-valid, provenance-bound, policy-checked candidate. Validation is necessary but not sufficient for action.
3. `COMMITTED_BELIEF`: durable belief committed by the microkernel with canonical serialization, digest, audit record, and restart reconciliation.
4. `AUTHORIZED_PLAN`: bounded plan authorized by the microkernel against current beliefs, policies, consent, capability registry, and effect budget.
5. `EXECUTED_EFFECT`: externally observable effect published through an authorized effect port with audit evidence.

## One-authority rule

Only the cognitive microkernel may commit beliefs, authorize plans, publish effects, commit memory, terminalize an episode, or advertise a capability. Adapters preserve backward compatibility by translating legacy outputs into `UNTRUSTED_PROPOSAL` or `VALIDATED_CANDIDATE`; they never become a second authority.

## Trust-boundary rules

- Internal LLMs and OpenAI are language interfaces only.
- Raw LLM text is never executable semantics, evidence, policy, memory mutation, or tool authority.
- Graphix output must be validated and authorized before execution or effect publication.
- CSIU and learning output are proposal-only until governed promotion by the microkernel.
- Source changes in a live serving container are prohibited; deploy a new image/commit instead.
- Every authoritative persistent mutation must have a single owner, transaction boundary, canonical digest, and fail-closed reconciliation.

## Current owner and transaction boundary

At this prompt's baseline, ADR-003 identifies `RuntimeContainer` and its `CognitiveKernel` as the current production owner. The exact transaction boundary for this architectural decision is the git commit containing these ADRs and `ami-invariants.yaml`; runtime durable transaction code is out of scope for P00 and must cite these invariants when implemented.
