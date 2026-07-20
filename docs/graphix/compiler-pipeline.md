# Graphix compiler pipeline

The Graphix compiler is a staged trust boundary from untrusted dialect artifacts to validated candidates and bounded human projections. The compiler never commits beliefs, authorizes plans, executes tools, or treats raw model text as authority.

Authoritative owner: `vulcan.graphix.validation.validate_graphix` owns compiler ingress validation. Transaction boundary: an immutable `ValidatedGraphixArtifact` digest is the handoff to `vulcan.graphix.compilers.compile_graphix`; later kernel action is required for any committed belief or authorized plan.

Validation stages are mandatory and ordered: structure, identity, source grounding, ontology, reference integrity, temporal validity, privacy/consent, resource bounds, authority, epistemic integrity, deontic constraints, and executable capability admission. Each stage produces typed diagnostics and the pipeline verifies that the immutable input digest is unchanged after every stage.

Migrations are pure and content-addressed. A migration record includes source and target digests, and migrations that silently change authority level are rejected.

Unknown extensions that claim security, authorization, policy, evidence, execution, tool, or capability meaning are rejected unless a future explicit schema admits them.

Rollback: disable callers of the new compiler and keep Graphix artifacts at `UNTRUSTED_PROPOSAL`; no persistent state is mutated by this package.
