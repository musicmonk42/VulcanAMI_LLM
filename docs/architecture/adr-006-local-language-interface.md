# ADR 0006: governed language interface

**Status:** Accepted for deterministic-only serving; neural/provider activation deferred.

## Decision

The neuro-symbolic runtime is the sole authority.  Language adapters have two
separate, narrow contracts: `LanguageInputPort` in `vulcan.runtime.semantic`
produces an untrusted, source-grounded interpretation proposal; and
`LanguageOutputPort` in `vulcan.runtime.output` produces an untrusted,
reference-only render draft from a minimized projection.  No adapter receives a
case, world state, memory, tools, finalizer, deployment object, history, or raw
reasoning result.

The supported serving surface is NFC-normalized `und` bounded arithmetic and
strict rendering of computed, unknown, error, and clarification results.
Deterministic parsing and strict rendering are permanent baselines and
fallbacks.  `disabled` and `deterministic_only` are the only selectable modes;
both require no model files and make no network call.  Local and OpenAI modes
are deliberately not selectable until an approved candidate, exact-container
prerequisites, and separate integration decision exist.

A release verifier may inspect a fully bound offline artifact directory, but it
is not a trust root or a serving selector.  Human release approval, artifact
verification, canary, revocation, and rollback policy must be implemented as a
closed deployment state machine before neural activation.  Training and
promotion remain offline and cannot be invoked from serving.

## Threat controls

Strict proposal validation reconstructs expressions from Unicode-codepoint
spans.  The output firewall accepts only exact ordered claim/caveat/citation
references; no prose, links, instructions, or unknown fields are representable.
Adapter exceptions and rejected drafts use strict rendering.  Artifact
verification rejects duplicate keys, path escape, symlinks, digest mismatch,
unapproved promotion, and non-improving evaluations.

## Legacy and deferred scope

Raw generation, hybrid execution, provider-only routing, runtime downloads,
distillation, and self-improvement are not approved language interfaces and
must not be connected to the canonical runtime.  No new ontology operation,
locale, fluent prose, model, or remote backup is authorized by this ADR.  Each
requires a typed compiler/engine, ledger and firewall rules, threat tests,
bake-off evidence, and a subsequent decision.

## Offline data and tokenizer controls

Offline data records are digest-only, default-deny, source-policy-bound, and
limited to `input_proposal` or `untrusted_render_draft` targets for the current
`und` arithmetic surface.  Grouped locked splits reject template-family
leakage.  The offline tokenizer contract permits only an NFC, bounded, unique,
immutable vocabulary and special-token map; token offsets never authorize
source spans.  These controls do not authorize a tokenizer, dataset, or model
candidate for serving.
