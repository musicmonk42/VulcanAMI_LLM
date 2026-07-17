# Local language-release boundary (Sequence 6 preparation)

## Status

Serving integration is intentionally **blocked**.  The canonical runtime remains
on `DeterministicLanguageInput` plus `render_strict`; this change adds no model
loader, provider, network client, environment-selected release, or runtime
selection.  The prerequisite security suite could not be collected in this
worktree because its root `tests/conftest.py` imports unavailable `numpy`.
Consequently, no claim is made that Sequences 1–5 are dependency-complete or
that a neural adapter is ready for activation.

## Offline verifier

`vulcan.local_language.verify_release()` is an offline release-process helper,
not a serving capability.  It accepts only a directory with `manifest.json` and
verifies all of the following before returning metadata:

* a strict, duplicate-key-free `local-language-release/1` manifest;
* one fixed role (`input-language-adapter` or `output-language-adapter`);
* one explicitly approved release identifier and human approval identifier;
* a complete fixed artifact set: weights, tokenizer, configuration, and the
  evaluation report, each contained under the release root and SHA-256-bound;
* a license identifier, runtime ABI, deterministic baseline, and a finite
  evaluation record whose candidate score is strictly better than its baseline
  and whose zero-tolerance result is true.

The verifier neither verifies a signature nor establishes provenance, license
validity, safety, quality, compliance, or deployment readiness.  Those claims
require separate evidence.  It never downloads, deserializes, loads, executes,
or tokenizes a model.

## Production reachability snapshot

| Symbol/path | Owner/caller | Inputs and output | Authority/capabilities | Docker-reachable disposition |
|---|---|---|---|---|
| `vulcan.runtime.container.RuntimeContainer.new` | Runtime lifecycle | deployment -> `CognitiveKernel` | owns canonical world state, governed memory, finalizer | reachable; unchanged |
| `vulcan.runtime.kernel.CognitiveKernel.handle` | Canonical ASGI route | server-created `Utterance` -> `KernelResult` | validates semantic proposal, executes kernel, finalizes response | reachable; strict renderer unchanged |
| `vulcan.runtime.semantic.DeterministicLanguageInput` | Kernel default | `Utterance` -> `InterpretationProposal` | bounded arithmetic proposal only | reachable baseline |
| `vulcan.runtime.semantic.render_strict` | Kernel | validated IR/ledger -> `RenderArtifact` | trusted deterministic renderer | reachable baseline |
| `vulcan.runtime.output.SemanticFirewall` | not yet kernel-owned | projection/draft -> result | validates structured references only | not activated |
| `vulcan.local_language.verify_release` | offline release process only | release directory -> metadata | reads bounded local files and hashes them; no network/model load | not imported by runtime |

A later integration may use only a verified, independently promoted release and
must first prove every Sequence 1–5 gate in an exact dependency-complete
container.  It must retain deterministic parser/renderer fallbacks and route
all model data through the existing typed semantic and output firewalls.
