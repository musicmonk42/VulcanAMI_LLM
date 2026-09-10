# Live transition permits and durable receipts

## Threat model

Python objects cannot defend against arbitrary malicious code in the same
interpreter. The mechanism here protects only Vulcan's supported, import-isolated
Phase-A trusted computing base. Models, plugins, tools, and every other future
untrusted component must execute out of process with an OS-enforced boundary.
Reflection or memory inspection by malicious in-process code is explicitly out of
scope and must never be represented as prevented.

## Live authority

`Principal` and `PrincipalKind` are descriptive identity only. In particular,
constructing `SYSTEM_KERNEL`, evidence, grants, or legacy command-authority values
grants no journal mutation capability. The private mutation port issues an
edge-specific permit for admission, validation, epistemic commitment, or
publication. A permit binds the issuing process instance, random nonce, bounded
expiry, actor, episode, predecessor head, snapshot, policy, verifier, and qualified
release. Exact-type and issuer-identity checks reject construction and cross-owner
replay. It cannot be copied or pickled and is consumed once before its unit of work
can commit.

The private module is restricted to the constitutional transaction owner and
trusted runtime kernel. Static qualification prohibits language, Graphix, and
reasoner packages from importing or naming it. This language-level isolation is a
TCB discipline, not a hostile-code sandbox.

## Durable evidence

A live permit is never persisted. Each committed successor instead records a
`vulcan-transition-receipt/1` content-addressed artifact in the same journal
transaction. It binds actor and credential provenance digests, command, episode,
predecessor and resulting heads, snapshot, policy and constitution evidence,
exact artifact digests, operation, verifier/trust-root identity, qualified release,
commit sequence, commit time, and permit issue/expiry evidence. Historical expiry
is checked against recorded commit time. The artifact is referenced by the
transactional receipt hash chain and cannot be submitted as a live permit.

No cryptographic signature or attestation is claimed. The receipt has an unsigned
content digest and journal hash-chain integrity. If external signing is introduced,
it must use an operator-provisioned stable root and retain public verification-key
history across rotation; a process-local secret is prohibited.

## Compatibility removal

Generic promotion, grants, and `CommandAuthority` remain only for non-journal
compatibility tests and legacy stores. The Phase-A journal path rejects them at its
exact live-permit boundary. Remove the compatibility adapter after all non-journal
callers migrate; do not extend it to new serving behavior.
