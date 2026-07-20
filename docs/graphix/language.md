# Graphix Language v1

Graphix Language v1 is the grounded ingress dialect for utterances and realization references. Its authoritative owner is the Graphix cognitive microkernel at the current repository transaction; language providers may propose meaning candidates but never author facts, effects, policy, memory mutation, tool authority, executable code, or final answers.

## Contracts and invariants

- `UtteranceRef` stores NFC-normalized text, locale, privacy label, and a full `sha256:` digest.
- `SourceSpan` uses Unicode-codepoint offsets over the normalized utterance and reconstructs exactly before use.
- `GroundedValue` requires at least one exact source span or a declared `SourceReference` external evidence source.
- `InterpretationCandidate` carries one or more dialogue acts, entity mentions, and semantic frames; `AmbiguityReport` preserves multiple candidates without selecting one.
- `SemanticFrame` explicitly preserves negation, modality, quantities, temporal expressions, and participant references.
- Provider proposals are scanned for forbidden fields including answer, belief, evidence, tool calls, policy, authorization, memory mutation, command, code, plan, and secrets.
- `external_provider_projection` emits digest and span metadata while redacting PERSONAL, SENSITIVE_PERSONAL, and SECRET span text.

## Compatibility adapter

`from_runtime_utterance` and `from_runtime_proposal` are the single compatibility adapter from the existing `vulcan.runtime.semantic` `Utterance` and `InterpretationProposal` span-only contracts. The adapter reconstructs expressions from spans rather than trusting provider text, so deterministic arithmetic ingress compiles into Graphix Language without granting executable authority.

## Migration and rollback

Migration is additive: start emitting Graphix Language alongside existing semantic ingress bundles, then switch readers to the new dialect. Rollback is to stop producing `graphix.language/1`; no persistent state transition or transaction boundary is introduced by this dialect.

## Security and privacy impact

The dialect fails closed for ungrounded semantic values, invalid spans, malformed confidence values, and forbidden provider authority fields. Privacy labels are explicit on utterances and spans, and external-provider projections redact sensitive source text while retaining auditable offsets and digests.
