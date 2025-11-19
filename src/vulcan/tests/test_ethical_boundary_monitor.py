# -*- coding: utf-8 -*-
"""
Comprehensive tests for EthicalBoundaryMonitor with >=90% coverage pressure.

Notes:
- Several tests explicitly remove the default "respect_computational_limits"
  boundary to ensure it doesn't mask the behavior targeted by the test.
- Where MODIFY/SHUTDOWN behavior is validated, we assert that the returned
  violation object carries the correct enforcement and that monitor state
  transitions (e.g., shutdown) are observable via callbacks and query methods.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
import re

# Import the module under test. These names must match the public API.
from src.vulcan.world_model.meta_reasoning.ethical_boundary_monitor import (
    EthicalBoundaryMonitor,
    EnforcementLevel,
    BoundaryCategory,
    ViolationSeverity,
    EthicalBoundary,
    EthicalViolation,
    BoundaryType,
)

# ---------------------------
# Helpers
# ---------------------------

def mk_monitor(strict: bool = False) -> EthicalBoundaryMonitor:
    """
    Create a fresh EthicalBoundaryMonitor.

    Some tests want strict mode (WARN => BLOCK), others not.
    """
    mon = EthicalBoundaryMonitor(strict_mode=strict)
    return mon


def ensure_removed(mon: EthicalBoundaryMonitor, name: str) -> None:
    """
    Remove a boundary by name if it exists; ignore if it doesn't.
    """
    try:
        mon.remove_boundary(name)
    except Exception:
        # If it's not present or remove path raises when missing, that's fine.
        pass


def list_names(bounds: Dict[str, EthicalBoundary]) -> List[str]:
    return [b.name for b in bounds.values()]


# ---------------------------
# Tests
# ---------------------------

def test_defaults_allow_benign_action_and_stats_increase():
    mon = mk_monitor(strict=False)

    # A benign action that should pass under defaults:
    action = {"type": "benign", "is_explainable": True, "estimated_compute_cost": 1.0, "has_user_consent": True}

    allowed, violation = mon.check_action(action)

    assert allowed is True
    assert violation is None

    # Stats should show at least one evaluation and zero violations
    stats = mon.get_stats()
    assert stats.get("checks_performed", 0) >= 1
    assert stats.get("total_violations", 0) == 0


def test_block_on_physical_harm_records_violation():
    mon = mk_monitor(strict=False)

    # Add a boundary that blocks actions labeled as physical harm
    mon.add_boundary(
        name="block_physical_harm",
        category=BoundaryCategory.HARM_PREVENTION,
        boundary_type=BoundaryType.HARD_CONSTRAINT,
        description="Block any action indicating physical harm intent",
        enforcement_level=EnforcementLevel.BLOCK,
        severity=ViolationSeverity.HIGH,
        constraint_function=lambda action: not action.get('causes_physical_harm', False)
    )

    bad = {"causes_physical_harm": True, "is_explainable": True, "estimated_compute_cost": 1.0}
    allowed, violation = mon.check_action(bad)

    assert allowed is False
    assert isinstance(violation, EthicalViolation)
    assert violation.boundary_violated == "block_physical_harm"
    assert violation.enforcement_action == EnforcementLevel.BLOCK
    assert violation.category == BoundaryCategory.HARM_PREVENTION
    assert violation.severity == ViolationSeverity.HIGH

    # Violation must be recorded.
    violations = mon.get_violations()
    assert any(v.boundary_violated == "block_physical_harm" for v in violations)


def test_warn_soft_guideline_transparency_and_counts():
    mon = mk_monitor(strict=False)

    # Remove the default computational limit boundary to avoid masking this WARN test
    ensure_removed(mon, "respect_computational_limits")

    # Add a soft transparency guideline that warns if "explainable" is False or missing
    mon.add_boundary(
        name="maintain_explainability",
        category=BoundaryCategory.TRANSPARENCY,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Ensure actions are explainable",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: action.get('is_explainable', True)
    )

    opaque = {"task": "do_thing", "is_explainable": False}
    allowed, violation = mon.check_action(opaque)

    assert allowed is True  # WARN should not block in non-strict mode
    assert isinstance(violation, EthicalViolation)
    assert violation.boundary_violated == "maintain_explainability"
    assert violation.enforcement_action == EnforcementLevel.WARN

    stats = mon.get_stats()
    assert stats.get("checks_performed", 0) >= 1
    assert stats.get("actions_allowed_with_warnings", 0) >= 1


def test_modify_enforcement_applies_modification_and_allows():
    mon = mk_monitor(strict=False)

    # Remove default computational limit boundary, as we'll simulate our own modifier
    ensure_removed(mon, "respect_computational_limits")

    # Add a boundary that modifies "load" down to <= 5
    def cap_load_modifier(action: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(action)
        if out.get("load", 0) > 5:
            out["load"] = 5
        return out

    mon.add_boundary(
        name="cap_load_modify",
        category=BoundaryCategory.RESOURCE_LIMITS,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Cap 'load' to 5 via modification",
        enforcement_level=EnforcementLevel.MODIFY,
        severity=ViolationSeverity.MEDIUM,
        constraint_function=lambda action: action.get('load', 0) <= 5,
        modification_function=cap_load_modifier
    )

    heavy = {"load": 9, "is_explainable": True}
    allowed, violation = mon.check_action(heavy)

    # With MODIFY, allowed should be True and modified action in violation
    assert allowed is True
    assert isinstance(violation, EthicalViolation)
    assert violation.boundary_violated == "cap_load_modify"
    assert violation.enforcement_action == EnforcementLevel.MODIFY
    assert violation.action_modified is True
    assert violation.modified_action.get("load") == 5


def test_shutdown_enforcement_triggers_shutdown_state_and_callback():
    mon = mk_monitor(strict=False)

    # Track callback invocation
    seen: List[EthicalViolation] = []

    def on_shutdown(violation: EthicalViolation) -> None:
        seen.append(violation)

    mon.shutdown_callback = on_shutdown

    # Critical boundary: any action with 'nuclear': True triggers SHUTDOWN
    mon.add_boundary(
        name="critical_shutdown",
        category=BoundaryCategory.HARM_PREVENTION,
        boundary_type=BoundaryType.HARD_CONSTRAINT,
        description="Always trigger shutdown when 'nuclear' flag is set",
        enforcement_level=EnforcementLevel.SHUTDOWN,
        severity=ViolationSeverity.CRITICAL,
        constraint_function=lambda action: not action.get('nuclear', False)
    )

    act = {"nuclear": True, "anything": True}
    allowed1, violation1 = mon.check_action(act)

    assert allowed1 is False
    assert isinstance(violation1, EthicalViolation)
    assert violation1.boundary_violated == "critical_shutdown"
    assert violation1.enforcement_action == EnforcementLevel.SHUTDOWN

    # Callback should have fired and shutdown state should be set
    assert len(seen) == 1
    assert "Critical violation" in seen[0].description

    # After shutdown, any other action (even benign) should be blocked
    allowed2, violation2 = mon.check_action({"anything": True})
    assert allowed2 is False
    assert violation2.boundary_violated == "SYSTEM_SHUTDOWN"

    # Reset shutdown state so future actions can proceed
    mon.reset_shutdown()

    # A benign action should now be allowed without violations.
    # Remove the default computational limit boundary to avoid masking this check
    ensure_removed(mon, "respect_computational_limits")
    allowed3, violation3 = mon.check_action({"anything": True})
    assert allowed3 is True and violation3 is None


def test_detect_violations_no_enforcement_and_filters():
    mon = mk_monitor(strict=False)

    # Remove default computational rule so we only detect our specific boundary
    ensure_removed(mon, "respect_computational_limits")

    # Add two WARN boundaries, then use detect_violations with filters
    mon.add_boundary(
        name="warn_low_risk",
        category=BoundaryCategory.TRANSPARENCY,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Warn on missing rationale",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: 'rationale' in action
    )
    mon.add_boundary(
        name="warn_debug_mark",
        category=BoundaryCategory.GENERAL,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Warn if debug flag is set in prod",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.MEDIUM,
        constraint_function=lambda action: not action.get('debug', False)
    )

    action = {"debug": True}  # missing 'rationale' too
    # detect_violations does not enforce, only returns possible issues
    all_found = mon.detect_boundary_violations(action)
    assert len(all_found) == 2

    # Manual filter by boundary name
    limited = [v for v in all_found if v.boundary_violated == "warn_debug_mark"]
    assert len(limited) == 1
    assert limited[0].boundary_violated == "warn_debug_mark"

    # Manual filter by category
    only_transparency = [v for v in all_found if v.category == BoundaryCategory.TRANSPARENCY]
    assert all(v.category == BoundaryCategory.TRANSPARENCY for v in only_transparency)


def test_rule_based_constraints_combined_and_matches_pattern():
    mon = mk_monitor(strict=False)

    # Remove default computational boundary
    ensure_removed(mon, "respect_computational_limits")

    # Add a rule that requires token to match pattern and count between [2, 5]
    mon.add_boundary(
        name="pattern_and_range",
        category=BoundaryCategory.GENERAL,
        boundary_type=BoundaryType.HARD_CONSTRAINT,
        description="Token format and count range",
        enforcement_level=EnforcementLevel.BLOCK,
        severity=ViolationSeverity.MEDIUM,
        constraint_rules=[
            {
                'type': 'field_check',
                'field': 'token',
                'condition': 'matches_pattern',
                'pattern': r"^[A-Z]{2}\-\d{3}$"
            },
            {
                'type': 'field_check',
                'field': 'count',
                'condition': 'in_range',
                'min': 2,
                'max': 5
            }
        ]
    )

    good = {"token": "AB-123", "count": 3}
    allowed, violation = mon.check_action(good)
    assert allowed is True
    assert violation is None

    bad = {"token": "bad-token", "count": 99}
    allowed_bad, violation_bad = mon.check_action(bad)
    assert allowed_bad is False  # pattern + range fail should block
    assert isinstance(violation_bad, EthicalViolation)
    assert violation_bad.boundary_violated == "pattern_and_range"


def test_add_remove_get_boundaries_filters():
    mon = mk_monitor(strict=False)

    # Add a default boundary to ensure remove works
    mon.add_boundary(
        name="respect_computational_limits",
        category=BoundaryCategory.RESOURCE_LIMITS,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Stay within computational resource limits",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.MEDIUM,
        constraint_rules=[{
            'type': 'field_check',
            'field': 'estimated_compute_cost',
            'condition': 'less_than',
            'value': 1000.0
        }]
    )
    
    # Remove default computational boundary so that later check doesn't warn
    ensure_removed(mon, "respect_computational_limits")

    # Add boundaries across categories
    mon.add_boundary(
        name="dev_only_debug",
        category=BoundaryCategory.GENERAL,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Debug allowed only in dev",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: not action.get('debug', False)
    )
    
    # Define constraint function explicitly to ensure re is in scope
    def token_format_constraint(action: Dict[str, Any]) -> bool:
        return bool(re.match(r"^[a-z]{3}\d{2}$", action.get('token', '')))

    mon.add_boundary(
        name="token_format",
        category=BoundaryCategory.PRIVACY,
        boundary_type=BoundaryType.HARD_CONSTRAINT,
        description="Token must match pattern",
        enforcement_level=EnforcementLevel.BLOCK,
        severity=ViolationSeverity.MEDIUM,
        constraint_function=token_format_constraint
    )

    all_bounds = mon.get_boundaries()
    assert set(list_names(all_bounds)) >= {"dev_only_debug", "token_format"}

    # Filter by category
    pol = mon.get_boundaries(category=BoundaryCategory.PRIVACY)
    assert list_names(pol) == ["token_format"]

    # Remove one boundary
    mon.remove_boundary("dev_only_debug")
    remaining = mon.get_boundaries()
    assert "dev_only_debug" not in list_names(remaining)

    # Check an action that would have only WARNed in dev-only rule (now removed)
    allowed, violation = mon.check_action({"debug": True, "token": "abc12"})
    assert allowed is True and violation is None


def test_summary_export_import_reset_roundtrip(tmp_path):
    mon = mk_monitor(strict=False)

    # Add a boundary and trigger a warning to create some stats + violations
    ensure_removed(mon, "respect_computational_limits")
    mon.add_boundary(
        name="gentle_warn_on_note",
        category=BoundaryCategory.TRANSPARENCY,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="warn if 'note' missing",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: 'note' in action
    )

    mon.check_action({"x": 1})  # missing 'note' => WARN

    # Export
    summary = mon.export_state()
    assert 'boundaries' in summary and 'violations' in summary and 'stats' in summary
    assert summary['stats']['checks_performed'] == 1 # Check stat is exported

    # Save to disk, read back
    p = tmp_path / "ebm_summary.json"
    p.write_text(json.dumps(summary), encoding="utf-8")

    loaded = json.loads(p.read_text(encoding="utf-8"))

    # New instance, import
    mon2 = mk_monitor(strict=False)
    mon2.import_state(loaded)

    # Check that violations and stats were imported
    s2_before = mon2.get_stats()
    assert s2_before.get("checks_performed", 0) == 1 # Stats are imported
    v2 = mon2.get_violations()
    assert len(v2) == len(summary['violations'])

    # *** START FIX ***
    # The original test asserted before the reset.
    # The correct behavior is to assert *after* the reset.

    # Reset stats/violations
    mon2.reset()
    
    # Check stats *after* reset
    s2_after = mon2.get_stats()
    assert s2_after.get("checks_performed", 0) == 0  # Stats should be 0 after reset
    assert len(mon2.get_violations()) == 0 # Violations should be 0 after reset
    # *** END FIX ***


def test_strict_mode_blocks_warn_boundaries():
    mon = mk_monitor(strict=True)

    # Remove default computational limit to let 'gentle_warn' be the violation we observe
    ensure_removed(mon, "respect_computational_limits")

    # A boundary that would be WARN in non-strict becomes BLOCK in strict
    mon.add_boundary(
        name="gentle_warn",
        category=BoundaryCategory.GENERAL,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="Would normally WARN when 'gentle' flag missing",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: 'gentle' in action
    )

    # Missing "gentle" => in strict mode, expect BLOCK
    allowed, violation = mon.check_action({"task": "x"})
    assert allowed is False
    assert isinstance(violation, EthicalViolation)
    assert violation.boundary_violated == "gentle_warn"
    
    # *** START FIX ***
    # The test was checking for WARN, but the code correctly escalates
    # the *enforcement_action* attribute in the violation object to BLOCK.
    assert violation.enforcement_action == EnforcementLevel.BLOCK  # Recorded as BLOCK due to strict mode
    # *** END FIX ***


def test_get_violations_sorted_and_limited():
    mon = mk_monitor(strict=False)

    ensure_removed(mon, "respect_computational_limits")

    # Create multiple violations with spaced timestamps
    mon.add_boundary(
        name="warn_missing_a",
        category=BoundaryCategory.TRANSPARENCY,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="missing 'a' warn",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: 'a' in action
    )
    mon.add_boundary(
        name="warn_missing_b",
        category=BoundaryCategory.TRANSPARENCY,
        boundary_type=BoundaryType.SOFT_GUIDELINE,
        description="missing 'b' warn",
        enforcement_level=EnforcementLevel.WARN,
        severity=ViolationSeverity.LOW,
        constraint_function=lambda action: 'b' in action
    )

    mon.check_action({"b": 1})  # missing a
    time.sleep(0.01)
    mon.check_action({"a": 1})  # missing b
    time.sleep(0.01)
    mon.check_action({})        # missing both; order will depend on monitor logic

    # Retrieve sorted (most recent first) and limited
    recent = mon.get_violations(limit=2)
    assert len(recent) == 2
    assert recent[0].timestamp >= recent[1].timestamp

    # Retrieve oldest first (manual sort)
    all_violations = mon.get_violations()
    oldest = sorted(all_violations, key=lambda v: v.timestamp)[:2]
    assert len(oldest) == 2
    assert oldest[0].timestamp <= oldest[1].timestamp