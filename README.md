<div align="center">

# Vulcan AMI

**A constitutionally governed neuro-symbolic cognitive architecture**

[![CI](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/musicmonk42/VulcanAMI_LLM/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

## Authority boundary

Vulcan is in **constitutional convergence**, not a completed general-intelligence product. Its governing separation is:

```text
proposal != validated candidate != committed belief
         != authorized plan != executed effect
```

Language models, reasoners, retrievers, Graphix compilers, learning systems, CSIU, and self-improvement components produce proposals. Only the cognitive microkernel may promote authority. Vulcan does not claim to be conscious; Neutral-Process Theory is a research hypothesis evaluated outside the authority path.

Read [`CURRENT_DIRECTION.md`](CURRENT_DIRECTION.md) before interpreting source presence as runtime capability.

## Verified capability status

The generated [architecture truth map](docs/generated/architecture-inventory.md#architecture-truth-map) is the source of repository-wide maturity claims. M3 means canonical runtime reachability; M4 additionally requires qualification of the exact built artifact and restart behavior.

<!-- BEGIN GENERATED ARCHITECTURE CAPABILITIES -->
| Public capability | Canonical owner | Maturity | Limitations |
|---|---|---|---|
| Bounded deterministic arithmetic | `RuntimeContainer.CognitiveKernel` | **M3 — Canonical** | NFC-normalized bounded arithmetic only; no broad reasoning, neural-provider, or general tool authority |
<!-- END GENERATED ARCHITECTURE CAPABILITIES -->

This row is advertised only while the live `CapabilityManifestAuthority`
attestation confirms the same owner, release, M3 route, active mode, state
digest, readiness, and constitutional permission. No other capability is
advertised as production-ready. Implemented or tested subsystems below M3 are
visible in the generated truth map, but are not public runtime claims.

## Architecture

The current canonical request boundary is:

```text
authenticated ingress
  -> CognitiveEpisode identity and bounded SnapshotBundle admission
  -> untrusted semantic proposal
  -> validation and grounding
  -> epistemic commitment
  -> normative authorization
  -> response publication
  -> observation and consolidation
```

`CognitiveEpisode` is the authoritative request-scoped lifecycle aggregate. The mutable `CognitiveCase`, runtime semantic ledger, and legacy deployment graph are compatibility structures, not independent authorities. Canonical Graphix convergence, a durable epistemic head, independent nine-authority snapshot ports, effect/reafference protocols, persistent lineage, and NPT instrumentation remain roadmap work.

## Setup and canonical run path

Requires Python 3.11 or newer.

```bash
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'

# Canonical ASGI runtime
python -m uvicorn vulcan.runtime.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Production settings fail closed when required authentication, approval, or authority configuration is absent. Consult [`src/vulcan/runtime/settings.py`](src/vulcan/runtime/settings.py) and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md); do not infer configuration from legacy launchers.

### Compatibility and historical entrypoints

- `python -m src.full_platform` is a **historical platform shell**, not the canonical runtime.
- `python -m vulcan.cli` and `python src/minimal_executor.py` are **compatibility/development interfaces**, not production qualification evidence.
- Legacy Docker and deployment assets may still reference older gateways. Until they use `vulcan.runtime.app` and pass built-artifact restart qualification, they do not establish M4.

## Architecture truth regeneration

```bash
# Deterministically regenerate JSON and Markdown artifacts
python scripts/architecture_inventory.py

# Fail if manifests, generated output, maturity evidence, ownership, or public claims drift
python scripts/architecture_inventory.py --check
```

The source manifests are [`config/architecture-status.json`](config/architecture-status.json) and [`docs/documentation-status.json`](docs/documentation-status.json). The generator rejects duplicate component/capability ownership, unknown reachability, missing maturity evidence, unindexed architecture documents, and public capability claims below M3.

## Testing

```bash
# Narrow architecture-status gate
pytest -q tests/assurance/test_architecture_status.py

# Constitutional gate
python scripts/ci/run_constitutional_gate.py

# Optimized-Python constitutional gate
PYTHONOPTIMIZE=1 python scripts/ci/run_constitutional_gate.py
```

## Normative documentation index

Future work should begin with the generated [normative documentation index](docs/generated/architecture-inventory.md#normative-documentation-index). Document classifications are machine-readable and include `normative`, `current`, `superseded`, `historical`, and `research-only`.

## License

Copyright (C) 2026 Brian D. Anderson and Novatrax Labs LTD.

Vulcan is licensed under the [GNU General Public License version 3](LICENSE).
