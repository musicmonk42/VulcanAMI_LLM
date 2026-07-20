from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.snapshots import SnapshotRef, construct_snapshot_bundle, default_snapshot_ref, require_bundle_snapshot
from vulcan.microkernel.state_machine import EpisodeState, EpisodeTransitionError

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
KINDS = ("world", "self", "social", "normative", "domain", "memory", "capability", "csiu", "alignment")

def clock():
    return NOW

class Lease:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True

class Provider:
    def __init__(self, kind, digest=None, bad_digest=False):
        self.kind = kind
        self.lease = Lease()
        self.digest = digest or (kind.encode().hex()[:64].ljust(64, "0"))
        self.bad_digest = bad_digest
    def lease_snapshot(self, *, kind, episode_id, acquired_at, expires_at):
        if kind != self.kind:
            raise RuntimeError("authority slot mismatch")
        digest = "not-full" if self.bad_digest else self.digest
        return SnapshotRef(kind, digest, "test.v1", f"owner:{kind}", "1", acquired_at, acquired_at, expires_at, f"rel:{episode_id}"), self.lease

def providers():
    return [Provider(k) for k in KINDS]

def actor():
    return ActorBinding("user:1", "a" * 64, "microkernel")

def test_snapshot_bundle_is_immutable_and_canonical():
    ps = providers()
    bundle = construct_snapshot_bundle(episode_id="ep1", providers=ps, clock=clock, lifetime=timedelta(minutes=5))
    assert bundle.digest == bundle.bundle_ref().state_digest
    assert [r["kind"] for r in bundle.to_json()["refs"]] == list(KINDS)
    with pytest.raises(FrozenInstanceError):
        bundle.bundle_id = "changed"


def test_mixed_snapshot_rejection_requires_explicit_rebase_event():
    bundle = construct_snapshot_bundle(episode_id="ep1", providers=providers(), clock=clock, lifetime=timedelta(minutes=5))
    ep = CognitiveEpisode.create(actor=actor(), request_id="r", input_digest="c" * 64, snapshot_bundle=bundle.bundle_ref(), clock=clock)
    with pytest.raises(EpisodeTransitionError, match="mixed snapshot"):
        ep.transition(EpisodeState.INTERPRETED, reason="normal", authority="microkernel", snapshot_ids=["9" * 64], clock=clock)
    ep.transition(EpisodeState.INTERPRETED, reason="explicit rebase to new bundle", authority="microkernel", snapshot_ids=["9" * 64], clock=clock)


def test_lease_cleanup_releases_all_pins_once():
    ps = providers()
    bundle = construct_snapshot_bundle(episode_id="ep1", providers=ps, clock=clock, lifetime=timedelta(minutes=5))
    bundle.close(); bundle.close()
    assert all(p.lease.closed for p in ps)
    with pytest.raises(RuntimeError, match="released"):
        bundle.validate_active(NOW)


def test_policy_domain_update_during_active_episode_keeps_old_leases_pinned():
    ps = providers()
    bundle = construct_snapshot_bundle(episode_id="ep1", providers=ps, clock=clock, lifetime=timedelta(minutes=5))
    old_domain = bundle.domain.digest
    ps[4].digest = "f" * 64
    newer = construct_snapshot_bundle(episode_id="ep2", providers=ps, clock=clock, lifetime=timedelta(minutes=5))
    assert bundle.domain.digest == old_domain
    assert newer.domain.digest == "f" * 64
    require_bundle_snapshot(bundle, old_domain, now=NOW)
    with pytest.raises(RuntimeError, match="mixed snapshot"):
        require_bundle_snapshot(bundle, newer.domain.digest, now=NOW)


def test_expiry_and_max_lifetime_fail_closed():
    bundle = construct_snapshot_bundle(episode_id="ep1", providers=providers(), clock=clock, lifetime=timedelta(minutes=1))
    with pytest.raises(RuntimeError, match="expired"):
        bundle.validate_active(NOW + timedelta(minutes=2))
    with pytest.raises(ValueError, match="bounded lifetime"):
        construct_snapshot_bundle(episode_id="ep1", providers=providers(), clock=clock, lifetime=timedelta(days=1))


def test_restart_reconciliation_rejects_digest_mismatch_and_malformed_authority():
    with pytest.raises(ValueError, match="full sha256"):
        construct_snapshot_bundle(episode_id="ep1", providers=[Provider(k, bad_digest=(k == "domain")) for k in KINDS], clock=clock, lifetime=timedelta(minutes=5))
    good = default_snapshot_ref("world", "owner", 1, {"rev": 1}, acquired_at=NOW, expires_at=NOW + timedelta(minutes=5), release_id="r")
    tampered = good.to_json(); tampered["digest"] = "0" * 64
    assert good.digest != tampered["digest"]
