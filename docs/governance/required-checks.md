# Required blocking checks

This repository treats CI as assurance evidence, not advisory reporting. Required branch protection checks should include:

- `dependency-light unit and contract evidence`
- `full integration evidence`
- `architecture fitness evidence`
- `static typing evidence`
- `lint and format evidence`
- `optimized Python parity evidence`
- `secret scan evidence`
- `SAST evidence`
- `dependency vulnerability policy evidence`
- `image E2E evidence`
- `supply-chain evidence`

Reports are only uploaded after their tool command succeeds; workflows must not fabricate empty reports or continue after required scanner/test failures. Optional research jobs must be labeled nonblocking and must not be configured as branch protection requirements.

## Action pin update policy

External GitHub Actions in required workflows are pinned by immutable commit SHA. Dependabot may open update PRs, but maintainers must verify the upstream tag-to-SHA mapping and require the same evidence gates before updating pins.

## Recommended CODEOWNERS

- Microkernel/runtime: `src/vulcan/runtime/`, `src/vulcan/core/`
- Safety and authorization: `src/vulcan/safety/`, `src/vulcan/runtime/auth.py`
- Constitution and governance: `docs/architecture/ami-invariants.yaml`, `docs/governance/`, `config/capabilities.yaml`
- CSIU and learning promotion: `src/vulcan/world_model/meta_reasoning/`, `src/vulcan/learning/`
- Release and supply chain: `.github/workflows/`, `docker/`, `helm/`, dependency lock files
