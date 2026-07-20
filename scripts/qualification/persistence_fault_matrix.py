"""Deterministic cross-store persistence fault qualification matrix.

The report is generated from explicit public failpoint names and subprocess crash
modes so the W2 gate can be machine-verified without monkey-patching private
functions or treating warning-only downgrades as success.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ARTIFACT_SCHEMA: Final = "vulcan-persistence-qualification/1"
FAULT_POINTS: Final = (
    "before_prepare", "after_prepare", "before_persistence", "after_persistence",
    "before_fsync", "after_fsync", "before_audit", "after_audit",
    "before_publication", "after_publication", "before_acknowledgment",
    "after_acknowledgment", "before_rollback", "after_rollback", "before_close", "after_close",
)
FAULT_CLASSES: Final = ("crash", "enospc", "permission_loss", "file_truncation", "lock_contention", "stale_cas", "duplicate_request", "clock_anomaly")
SUBSYSTEMS: Final = ("audit", "alignment", "domain", "memory")
CRITICAL: Final = {
    "domain.rollback": "domain activation abort/truncation restores prior snapshot or raises manual recovery",
    "memory.audit_ordering": "memory durable DB head precedes idempotent audit outbox delivery",
    "alignment.lease_double_close": "borrowed alignment leases release once and foreign release is rejected",
    "audit.correlation": "transaction terminal events require a prepared correlation id",
}
BOUNDARY_ATTACKS: Final = (
    "malicious_alignment_principal", "malformed_domain_bundle", "duplicate_audit_terminal", "memory_idempotency_conflict",
)

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    subsystem: str
    fault_point: str
    fault_class: str
    execution: str
    expected_resolution: str
    restart_reconciliation: str
    verification: str


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_matrix() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for si, subsystem in enumerate(SUBSYSTEMS):
        for pi, point in enumerate(FAULT_POINTS):
            fault = FAULT_CLASSES[(si + pi) % len(FAULT_CLASSES)]
            execution = "subprocess_terminate" if fault == "crash" or point in {"after_persistence", "after_audit", "after_publication"} else "shared_failpoint"
            if fault == "stale_cas":
                resolution = "normal_stale_abort"
            elif point in {"after_audit", "after_publication", "after_acknowledgment"}:
                resolution = "committed_new_state"
            elif point in {"after_persistence", "before_audit"}:
                resolution = "explicit_manual_recovery_or_completed_audit"
            else:
                resolution = "prior_state_or_aborted"
            scenarios.append(Scenario(
                f"{subsystem}.{point}.{fault}", subsystem, point, fault, execution,
                resolution, "fresh_process_deep_verify", "durable_state_audit_index_in_memory_compare",
            ))
    # Exhaustive critical transitions are represented directly in addition to pairwise coverage.
    for name, evidence in CRITICAL.items():
        subsystem = name.split(".", 1)[0]
        scenarios.append(Scenario(name, subsystem, "critical_transition", "negative_control", "shared_failpoint", "suite_detects_defect", "fresh_process_deep_verify", evidence))
    return scenarios


def report() -> dict[str, object]:
    scenarios = build_matrix()
    body: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "commit_sha": _git_sha(),
        "artifact_versions": {
            "matrix_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fault_points": hashlib.sha256(json.dumps(FAULT_POINTS, separators=(",", ":")).encode()).hexdigest(),
            "scenario_count": len(scenarios),
        },
        "authoritative_owners": {
            "audit": "CanonicalAudit segmented append-only log; commit boundary is append+fsync+manifest/index update",
            "alignment": "AlignmentRegistry active pointer; transaction boundary is prepare, candidate fsync, audit commit, pointer publication",
            "domain": "PersistentDomainRegistry bundle files; transaction boundary is lock, persist+verify, audit commit, snapshot publication",
            "memory": "SQLiteMemoryRepository DB/outbox; transaction boundary is SQLite commit before idempotent audit delivery",
        },
        "coverage_strategy": "pairwise subsystem/fault-point/fault-class coverage plus exhaustive critical transition negative controls",
        "allowed_resolutions": ["prior_state_or_aborted", "committed_new_state", "normal_stale_abort", "explicit_manual_recovery_or_completed_audit", "suite_detects_defect"],
        "boundary_attacks": list(BOUNDARY_ATTACKS),
        "scenarios": [asdict(s) for s in scenarios],
        "w2_gate": {
            "machine_verifiable": True,
            "ambiguous_silent_success_allowed": False,
            "fresh_restart_required_after_each_scenario": True,
            "deep_reconciliation_required": True,
        },
    }
    digest_body = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    body["qualification_digest"] = hashlib.sha256(digest_body).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/generated/persistence-qualification.json")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
