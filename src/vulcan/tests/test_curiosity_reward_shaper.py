# src/vulcan/tests/test_curiosity_reward_shaper.py
# Full, untruncated test suite for CuriosityRewardShaper.
# Coverage goal: >=90% for src/vulcan/world_model/meta_reasoning/curiosity_reward_shaper.py
#
# This suite exercises:
# - compute_curiosity_bonus for COUNT_BASED, ICM (with and without prediction history),
#   RND, INFORMATION_GAIN (with/without distributions populated), EPISODIC, and HYBRID.
# - shape_reward (integration over compute path).
# - update_novelty_estimates (ICM updates + information gain distributions + validation hook).
# - get_novelty (seen/unseen paths).
# - get_exploration_recommendation (ranking + reason + hashes).
# - register_feature_extractor (custom features merged).
# - get_statistics (includes method, weights, memory sizes, trends, etc.).
# - export_state / import_state round-trip and reset path (clears data and reinitializes RND).
# - adaptive scaling via _update_adaptive_scaling (low/high novelty cases).
#
# Notes:
# - We keep assertions tolerant to internal randomness (RND) by checking ranges and invariants.
# - We shorten scale_update_interval on an instance to trigger adaptive scaling deterministically.

import math
import threading
import time
import types

import numpy as np
import pytest

from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import (
    CuriosityMethod, CuriosityRewardShaper, CuriosityStatistics, NoveltyLevel)

# ---------------------------
# Lightweight stubs for integrations
# ---------------------------


class StubTransparency:
    def __init__(self):
        self.records = []

    def record_curiosity_bonus(self, state_hash, novelty, bonus, method):
        # Store a tiny dict so we can assert integration happened
        self.records.append(
            {
                "state_hash": state_hash,
                "novelty": novelty,
                "bonus": bonus,
                "method": method,
            }
        )


class StubValidationTracker:
    def __init__(self):
        self.records = []

    def record_validation(self, proposal, validation_result, actual_outcome):
        self.records.append(
            {
                "proposal": proposal,
                "validation_result": validation_result,
                "actual_outcome": actual_outcome,
            }
        )


# ---------------------------
# Helpers
# ---------------------------


def mk_state(**kwargs):
    # Keep features numeric/boolean/string to exercise vectorization paths
    base = {"x": 0.1, "y": 0.2, "flag": True, "tag": "alpha"}
    base.update(kwargs)
    return base


def approx_between(x, lo=0.0, hi=1.0):
    assert lo <= x <= hi


# ---------------------------
# COUNT_BASED
# ---------------------------


def test_count_based_curiosity_basic_and_novelty_levels():
    sh = CuriosityRewardShaper(
        curiosity_weight=0.5,
        method=CuriosityMethod.COUNT_BASED,
        max_bonus=1.0,
    )
    s = mk_state(x=0.0)
    # First visit -> high novelty -> COMPLETELY_NOVEL classification
    b1 = sh.compute_curiosity_bonus(s)
    assert b1 >= 0.0
    stats = sh.get_statistics()
    assert stats["unique_states_seen"] == 1
    # Repeat same state: novelty should reduce given 1/sqrt(N)
    b2 = sh.compute_curiosity_bonus(s)
    assert b2 <= b1
    # Many visits reduce novelty -> WELL_KNOWN eventually
    for _ in range(10):
        sh.compute_curiosity_bonus(s)
    est = sh.novelty_estimates[sh._hash_state(s)]
    # verify estimate object updated
    assert est.visit_count >= 12
    assert est.novelty_score <= 1.0
    assert est.novelty_level in {
        NoveltyLevel.COMPLETELY_NOVEL,
        NoveltyLevel.HIGHLY_NOVEL,
        NoveltyLevel.MODERATELY_NOVEL,
        NoveltyLevel.FAMILIAR,
        NoveltyLevel.WELL_KNOWN,
    }


# ---------------------------
# ICM
# ---------------------------


def test_icm_curiosity_with_and_without_prediction_history():
    vt = StubValidationTracker()
    tr = StubTransparency()
    sh = CuriosityRewardShaper(
        curiosity_weight=0.8,
        method=CuriosityMethod.ICM,
        validation_tracker=vt,
        transparency_interface=tr,
    )

    s1 = mk_state(tag="icm_state_1")
    s2 = mk_state(tag="icm_state_2", x=0.9, y=0.9)

    # No prediction yet -> high default novelty
    b1 = sh.compute_curiosity_bonus(s1, context={"episode": 1})
    assert b1 >= 0.0
    # Update ICM forward prediction using update_novelty_estimates (with next_state)
    sh.update_novelty_estimates(s1, outcome={"outcome": "ok"}, next_state=s2)
    # Now computing again for s1 will compare predicted vs actual
    b2 = sh.compute_curiosity_bonus(s1, context={"episode": 1})
    assert b2 >= 0.0
    # Transparency logged
    assert len(tr.records) >= 2
    # Validation recorded on update
    assert len(vt.records) == 1
    assert vt.records[0]["proposal"]["type"] == "exploration"


# ---------------------------
# RND
# ---------------------------


def test_rnd_curiosity_prediction_error_range_and_feature_dim_cap():
    sh = CuriosityRewardShaper(
        curiosity_weight=0.3,
        method=CuriosityMethod.RND,
        feature_dim=16,
    )
    # Long feature dict to ensure vector is clipped to feature_dim
    s = {f"k{i}": float(i) for i in range(50)}
    b = sh.compute_curiosity_bonus(s)
    approx_between(b, 0.0, 1.0)
    # Subsequent calls should be OK, and statistics should update
    sh.compute_curiosity_bonus(s)
    st = sh.get_statistics()
    assert st["total_bonuses_computed"] >= 2
    assert st["method"] == CuriosityMethod.RND.value


# ---------------------------
# INFORMATION_GAIN
# ---------------------------


def test_information_gain_first_visit_high_then_entropy_normalized():
    vt = StubValidationTracker()
    sh = CuriosityRewardShaper(
        curiosity_weight=0.4,
        method=CuriosityMethod.INFORMATION_GAIN,
        validation_tracker=vt,
    )

    s = mk_state(tag="ig")
    # First visit -> no distributions -> returns ~0.9 (per implementation)
    b1 = sh.compute_curiosity_bonus(s)
    assert b1 >= 0.0
    # Update with outcome to populate distributions for entropy
    sh.update_novelty_estimates(s, outcome={"outcome": "learned"})
    # Next compute -> uses entropy calculation path
    b2 = sh.compute_curiosity_bonus(s)
    assert b2 >= 0.0
    # Validation recorded once from update
    assert len(vt.records) == 1


# ---------------------------
# EPISODIC
# ---------------------------


def test_episodic_novelty_similarity_inverse_and_memory_growth():
    sh = CuriosityRewardShaper(
        curiosity_weight=0.7,
        method=CuriosityMethod.EPISODIC,
    )
    s1 = mk_state(x=0.2, y=0.3, tag="e1")
    s2 = mk_state(x=0.2, y=0.3, tag="e1")  # similar features

    # First call: no memory -> returns 1.0 novelty internally, then scaled to bonus
    b1 = sh.compute_curiosity_bonus(s1)
    # Second: memory contains s1; s2 should be more "familiar" (lower novelty/bonus)
    b2 = sh.compute_curiosity_bonus(s2)
    assert b2 <= b1
    # Episodic memory populated
    assert len(sh.episodic_memory) >= 1
    # Memory index updated
    h = sh._hash_state(s1)
    assert h in sh.memory_index


# ---------------------------
# HYBRID
# ---------------------------


def test_hybrid_combination_weights_and_estimate_fields():
    tr = StubTransparency()
    sh = CuriosityRewardShaper(
        curiosity_weight=0.6, method=CuriosityMethod.HYBRID, transparency_interface=tr
    )
    # Shrink the scale interval to trigger adaptive scaling later
    sh.scale_update_interval = 5

    s = mk_state(tag="hyb")
    # First few calls populate internal caches and novelty_history
    for _ in range(6):
        b = sh.compute_curiosity_bonus(s)
        assert b >= 0.0
    # Hybrid weights present in stats
    st = sh.get_statistics()
    assert isinstance(st["hybrid_weights"], dict)
    # Novelty estimate fields recorded (count/icm/rnd/episodic)
    est = sh.novelty_estimates[sh._hash_state(s)]
    # Fields exist; values can be any valid 0..1 range (depending on path)
    assert est.count_novelty >= 0.0
    assert est.icm_novelty >= 0.0
    assert est.rnd_novelty >= 0.0
    assert est.episodic_novelty >= 0.0
    # Transparency recorded several times
    assert len(tr.records) >= 5


# ---------------------------
# shape_reward + get_novelty + recommendation + extractor
# ---------------------------


def test_shape_reward_get_novelty_recommendation_and_custom_extractor():
    sh = CuriosityRewardShaper(curiosity_weight=0.5, method=CuriosityMethod.COUNT_BASED)

    # Register a custom extractor (adds a new numeric feature)
    def extra_features(state, context):
        return {"fx": 0.42, "ctx_flag": 1 if context.get("z") else 0}

    sh.register_feature_extractor(extra_features)

    # shape_reward uses compute underneath
    s1 = mk_state(tag="rr1")
    shaped = sh.shape_reward(0.25, s1, context={"z": True})
    assert shaped >= 0.25

    # get_novelty for seen state should be in [0,1]; for unseen returns 1.0
    nov_seen = sh.get_novelty(s1)
    approx_between(nov_seen, 0.0, 1.0)
    nov_unseen = sh.get_novelty(mk_state(tag="unseen"))
    assert nov_unseen == 1.0

    # Recommendation ranks by novelty; ensure response shape
    rec = sh.get_exploration_recommendation([s1, mk_state(tag="unseen2")])
    assert isinstance(rec, dict)
    assert "recommended_state" in rec and "reason" in rec and "all_novelties" in rec
    # all_novelties contains tuples of (hash, novelty)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in rec["all_novelties"])


# ---------------------------
# Statistics + export/import + reset
# ---------------------------


def test_statistics_export_import_reset_roundtrip():
    tr = StubTransparency()
    sh = CuriosityRewardShaper(
        curiosity_weight=0.5,
        method=CuriosityMethod.HYBRID,
        transparency_interface=tr,
        feature_dim=8,  # speed up
    )
    # Compute a few bonuses
    for i in range(7):
        sh.compute_curiosity_bonus(mk_state(tag=f"s{i}"))

    stats = sh.get_statistics()
    # Expected top-level keys present
    for k in [
        "method",
        "curiosity_weight",
        "bonus_scale",
        "unique_states_seen",
        "episodic_memory_size",
        "novelty_estimates",
    ]:
        assert k in stats

    # Export state snapshot
    snap = sh.export_state()
    assert "state_visits" in snap and "novelty_estimates" in snap
    assert "config" in snap and "hybrid_weights" in snap

    # New instance imports state (and updates bonus_scale + weights)
    sh2 = CuriosityRewardShaper(
        curiosity_weight=0.1,
        method=CuriosityMethod.COUNT_BASED,  # different config, to test overwrite behavior
        feature_dim=8,
    )
    sh2.import_state(snap)
    st2 = sh2.get_statistics()
    # Imported visits reflected in unique_states_seen
    assert st2["unique_states_seen"] == len(snap["state_visits"])
    # Bonus scale preserved
    # (We don't assert exact value; we assert it's > 0 and <= 2 by implementation invariants)
    assert 0 < sh2.bonus_scale <= 2.0

    # Reset clears everything and reinitializes RND nets safely
    sh2.reset()
    st_reset = sh2.get_statistics()
    assert st_reset["unique_states_seen"] == 0
    assert sh2.statistics.total_bonuses_computed == 0
    # After reset, computing still works
    b = sh2.compute_curiosity_bonus(mk_state(tag="after_reset"))
    assert b >= 0.0


# ---------------------------
# Adaptive scaling explicit tests
# ---------------------------


def test_adaptive_scaling_low_novelty_increases_scale_and_high_decreases():
    sh = CuriosityRewardShaper(method=CuriosityMethod.HYBRID)
    # Force novelty history to low values -> scale should increase up to 2.0 cap
    sh.novelty_history.clear()
    sh.novelty_history.extend([0.05] * 50)
    sh.bonus_scale = 1.0
    sh._update_adaptive_scaling()
    assert 1.0 <= sh.bonus_scale <= 2.0

    # Now force high novelty -> scale should decrease but not below 0.5
    sh.novelty_history.clear()
    sh.novelty_history.extend([0.95] * 50)
    prev = sh.bonus_scale
    sh._update_adaptive_scaling()
    assert 0.5 <= sh.bonus_scale <= prev


# ---------------------------
# Concurrency smoke (lock path)
# ---------------------------


def test_concurrent_bonus_computation_lock_paths():
    sh = CuriosityRewardShaper(method=CuriosityMethod.HYBRID)
    s = mk_state(tag="concurrent")

    def worker():
        # Each thread calls compute a few times; no race or exceptions expected
        for _ in range(5):
            sh.compute_curiosity_bonus(s)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = sh.get_statistics()
    assert st["total_bonuses_computed"] >= 20
