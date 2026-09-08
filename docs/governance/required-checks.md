# Required constitutional checks

## Assurance authority and transaction boundary

This change affects **assurance authority only**; it does not alter cognition. Previously,
workflow labels could stand in for checks they did not run, required Python jobs could use
the runner's ambient interpreter, and dependency installation was not one reproducible
transaction. The new evidence transaction starts with a pinned checkout and Python 3.11,
installs `requirements-constitutional.txt` with `--require-hashes`, runs a real check, and
writes evidence only after that command succeeds.

The locally reproducible transaction is:

```bash
python scripts/ci/run_constitutional_gate.py --bootstrap
```

The gate performs workflow anti-bypass/reproducibility lint, mypy over the constitutional
assurance scripts, Black and isort in check mode, the authoritative-episode integration
set, and core security contracts under normal and optimized Python. Once dependencies
are installed, core correctness uses no network, external model/provider, cloud
credentials, Redis, or Kubernetes.

## Required branch-protection contexts

A repository administrator must configure these GitHub check contexts for `main`:

- `constitutional unit and contract evidence`;
- `authoritative episode constitutional integration`;
- `architecture and workflow reproducibility`;
- `mypy constitutional assurance surface`;
- `Black and isort check mode`;
- `normal and optimized Python security parity`;
- the security evidence jobs in `blocking-security`;
- the image and runtime qualification jobs only after their runner capacity is verified.

Also require pull requests, require the branch to be current before merge, dismiss stale
approvals, prohibit force pushes/deletion, and restrict bypass permission. These are
manual GitHub settings: this committed document does **not** protect the branch or prove
that an account has runners, secrets, or branch-protection privileges.

## Compatibility and quarantine

`requirements-constitutional.txt` is the sole compatibility adapter retained for the
repository's broader historical dependency graph: it projects only the tools needed for
constitutional evidence from `requirements-hashed.txt`. Remove it when the canonical
package lock can install the same minimal assurance environment without research/runtime
extras.

`security-smoke.yml`, `scalability_test.yml`, the generic legacy deployment workflow,
and the Azure and Tencent deployment workflows are manual, explicitly advisory. They
must not be configured as required checks. Their removal condition is replacement by tests that
exercise the canonical runtime without provider or cloud authority. The runtime image
workflow remains qualification evidence, not proof of M4 until the exact artifact and
restart run have succeeded on configured GitHub runners.

## Unresolved limitations

The lightweight secret-pattern scanner carries explicit path exceptions for four findings
that predate this assurance change: three test-fixture files and the legacy
self-improvement development default.
Those explicit path exceptions are not a statement that the content is production-safe;
they prevent unrelated baseline debt from masquerading as a newly green scan. Remove
each exception when that legacy settings or self-improvement surface is repaired or
retired. GitHub-hosted runner availability and branch-protection configuration also
remain externally administered, so local success cannot qualify them.
