# Phase-A immutable artifact qualification

A09 is an artifact qualification gate, not a source-test label. It accepts one
pre-verified wheel digest and one `repository@sha256:<digest>` image subject.
Mutable tags, dirty source trees, changed wheels, pre-existing output files,
incomplete catalogs, omitted critical scenarios, and switching the image subject
are refused before a PASS bundle can be written.

`build_candidate_images.py` consumes the frozen wheel directly using
`Dockerfile.candidate`; it never rebuilds source. It requires an immutable base
image and an offline hash-locked wheelhouse, builds two untagged images, and
requires identical immutable image IDs plus normalized config/rootfs inspection.
The first identical ID is the sole candidate. Registry publication is outside
this procedure.

`qualify_phase_a_artifact.py` executes argv arrays without a shell and derives each
result only from the real exit status. The evidence bundle stores argv and
stdout/stderr digests, never self-authored scenario claims. It is written only if
all critical scenarios pass on the same immutable subject, including normal and
optimized installed runs, lifespan/routes, authentication and idempotency,
arithmetic publication, journal/audit recovery, authority counterexamples,
kill/restart, TCB manifest, SBOM, and non-root/read-only filesystem checks.

The bundle binds source revision and clean-tree proof, wheel and image digests,
build/runtime locks and bases, ActorBinding schema and vectors, journal and
receipt-chain schema, authority-verification mode/history, migration policy,
route/import manifests, test catalog, and harness. Capability evidence may be
redirected to this bundle only after the bundle exists and independently verifies.

This container does not provide Docker or Podman and registry access is denied.
Consequently image construction and the A09 critical catalog are `NOT_EXECUTED`;
no qualified digest or Phase-B admission bundle is emitted here. Source and wheel
checks do not substitute for that missing image evidence.
