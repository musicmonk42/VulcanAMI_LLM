# Instructions for AI coding agents

Read [`CURRENT_DIRECTION.md`](CURRENT_DIRECTION.md) before proposing or changing architecture.

## What Vulcan is

Vulcan is a governed neuro-symbolic cognitive architecture. The transformer is a language/proposal interface. It must never become the epistemic, normative, memory, policy, or effect authority.

The current engineering program is **constitutional convergence**. Do not treat the presence of a subsystem in source as proof that it is canonical, production-reachable, or authorized.

## Non-negotiable rules

- Preserve `proposal != belief != plan != effect`.
- `CognitiveEpisode` is the authoritative request-scoped lifecycle aggregate.
- Every live episode must be bound to one admitted multi-authority snapshot bundle.
- Only the cognitive microkernel may promote authority.
- Compatibility structures may project or translate state; they may not become a second authority.
- Raw LLM output is never evidence, executable semantics, policy, memory mutation, or tool authority.
- Every externally observable effect eventually requires intent, authorization, a scoped capability, an execution receipt, and observed consequences.
- Serving processes must not install their own source changes.
- Do not claim or encode that Vulcan is conscious. NPT is a falsifiable research hypothesis evaluated outside the authority path.

## Current implementation boundary

This branch begins moving the canonical request path onto immutable episodes and admitted state bundles. It does **not** complete:

- canonical Graphix convergence;
- the durable Graphix Epistemic ledger;
- audit derivation from episode transitions;
- independent production implementations of all nine state-authority ports;
- persistent agent lineage;
- effect/reafference protocols;
- causal autobiographical memory;
- NPT scientific instrumentation.

Those items are ordered in [`docs/roadmap/constitutional-convergence-plan.md`](docs/roadmap/constitutional-convergence-plan.md). Work in that order unless a prerequisite defect must be repaired first.

## Change discipline

For each architectural PR:

1. name the sole authority affected;
2. state the old and new transaction boundary;
3. identify compatibility code and its removal condition;
4. add failure, restart, replay, and concurrency tests where relevant;
5. update the roadmap maturity status;
6. avoid broad refactors that combine unrelated authority changes.

Do not add another framework or repository to escape integration work. Build the new constitutional spine inside Vulcan and progressively strangle legacy authority paths.
