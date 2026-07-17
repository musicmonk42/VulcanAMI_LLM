# ADR 0004: typed semantic ingress and response integrity

The Docker-selected runtime accepts text only as a request-scoped `Utterance`.
A `LanguageInputPort` can emit an untrusted, bounded `InterpretationProposal`;
validation constructs an `InterpretationBundle`, and only kernel selection creates
an `AcceptedInterpretation`. The initial supported ontology is restricted arithmetic.
Unsupported input produces an `unknown` claim rather than provider reasoning.

Claims and derivations are append-only request-scoped records. `ResponseIR` is
versioned and the strict renderer consumes only that IR. No fluent renderer is
present: this intentionally provides conservative fallback until a verifier exists.
Unicode spans use NFC-normalized Unicode code-point offsets. Canonical digests use
sorted UTF-8 JSON with non-finite values rejected. The raw utterance is not copied
into `CognitiveCase`; request-local digests are not represented as durable audits.

This replaces the legacy raw FastAPI request adapter on the production graph.
Rollback is a deployment rollback, not a compatibility route: reintroducing the
legacy adapter would violate this boundary.
