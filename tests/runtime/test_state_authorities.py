from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Barrier, Thread

import pytest

from vulcan.microkernel.snapshots import construct_snapshot_bundle
from vulcan.runtime.state_authorities import (
    ContentBoundStateAuthority,
    StateAuthoritySet,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
KINDS = (
    "world",
    "self",
    "social",
    "normative",
    "domain",
    "memory",
    "capability",
    "csiu",
    "alignment",
)


class Owner:
    def __init__(self, kind):
        self.kind, self.revision, self.value, self.releases = kind, 1, "empty", 0

    def read(self):
        value, revision = self.value, self.revision
        owner = self

        class Lease:
            def close(self):
                owner.releases += 1

        return str(revision), {"value": value}, Lease()


def authority_set(owners):
    values = [
        ContentBoundStateAuthority(
            kind=k,
            owner=f"owner:{k}",
            schema=f"{k}.v1",
            release="test-release",
            read=o.read,
        )
        for k, o in zip(KINDS, owners, strict=True)
    ]
    return StateAuthoritySet(*values)


def bundle(authorities, episode="case-1"):
    return construct_snapshot_bundle(
        episode_id=episode,
        providers=authorities.providers(),
        clock=lambda: NOW,
        lifetime=timedelta(minutes=5),
    )


@pytest.mark.parametrize("changed", range(9))
def test_each_authority_digest_is_isolated(changed):
    owners = [Owner(k) for k in KINDS]
    authorities = authority_set(owners)
    before = bundle(authorities).ref_digests()
    owners[changed].value = "changed"
    owners[changed].revision += 1
    after = bundle(authorities, "case-2").ref_digests()
    assert [
        i
        for i, pair in enumerate(zip(before, after, strict=True))
        if pair[0] != pair[1]
    ] == [changed]


@pytest.mark.parametrize(
    ("override", "later"),
    (
        ({"owner": "two"}, timedelta()),
        ({"schema": "world.v2"}, timedelta()),
        ({"release": "r2"}, timedelta()),
        ({}, timedelta(seconds=1)),
    ),
)
def test_state_identity_binds_owner_release_schema_and_validity(override, later):
    owner = Owner("world")
    base = ContentBoundStateAuthority(
        kind="world", owner="one", schema="world.v1", release="r1", read=owner.read
    )
    first, lease = base.lease_snapshot(
        kind="world",
        episode_id="case-1",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )
    options = {
        "kind": "world",
        "owner": "one",
        "schema": "world.v1",
        "release": "r1",
        "read": owner.read,
    }
    options.update(override)
    second, second_lease = ContentBoundStateAuthority(**options).lease_snapshot(
        kind="world",
        episode_id="case-1",
        acquired_at=NOW + later,
        expires_at=NOW + timedelta(minutes=1) + later,
    )
    assert first.digest != second.digest
    lease.close()
    second_lease.close()


def test_slot_mismatch_and_reader_failure_fail_closed():
    authority = ContentBoundStateAuthority(
        kind="world",
        owner="world",
        schema="world.v1",
        release="r",
        read=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    with pytest.raises(RuntimeError, match="cannot serve"):
        authority.lease_snapshot(
            kind="self",
            episode_id="case",
            acquired_at=NOW,
            expires_at=NOW + timedelta(1),
        )
    with pytest.raises(RuntimeError, match="unavailable"):
        authority.lease_snapshot(
            kind="world",
            episode_id="case",
            acquired_at=NOW,
            expires_at=NOW + timedelta(1),
        )


def test_invalid_state_releases_a_pin_acquired_inside_reader():
    owner = Owner("world")
    authority = ContentBoundStateAuthority(
        kind="world",
        owner="world",
        schema="world.v1",
        release="r",
        read=lambda: ("1", {"not-canonical": object()}, owner.read()[2]),
    )
    with pytest.raises((TypeError, ValueError)):
        authority.lease_snapshot(
            kind="world",
            episode_id="case",
            acquired_at=NOW,
            expires_at=NOW + timedelta(1),
        )
    assert owner.releases == 1


def test_cleanup_failure_does_not_mask_invalid_snapshot_state():
    class BrokenLease:
        def close(self):
            raise RuntimeError("cleanup failed")

    authority = ContentBoundStateAuthority(
        kind="world",
        owner="world",
        schema="world.v1",
        release="r",
        read=lambda: ("1", {"not-canonical": object()}, BrokenLease()),
    )
    with pytest.raises((TypeError, ValueError)) as excinfo:
        authority.lease_snapshot(
            kind="world",
            episode_id="case",
            acquired_at=NOW,
            expires_at=NOW + timedelta(1),
        )
    assert "snapshot lease cleanup failed" in " ".join(excinfo.value.__notes__)


def test_one_provider_cannot_be_aliased_across_authority_slots():
    owners = [Owner(k) for k in KINDS]
    authorities = authority_set(owners)
    values = list(authorities.providers())
    values[1] = values[0]
    with pytest.raises(RuntimeError, match="independent provider"):
        StateAuthoritySet(*values)


def test_concurrent_episodes_pin_independent_leases_and_release_once():
    owners = [Owner(k) for k in KINDS]
    authorities, barrier, bundles = authority_set(owners), Barrier(3), []

    def admit(index):
        barrier.wait()
        bundles.append(bundle(authorities, f"case-{index}"))
        barrier.wait()

    threads = [Thread(target=admit, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert len(bundles) == 2 and all(not item.released for item in bundles)
    for item in bundles:
        item.close()
        item.close()
    assert all(owner.releases == 2 for owner in owners)


def test_restart_replay_is_stable_but_expiry_is_enforced():
    owners = [Owner(k) for k in KINDS]
    first = bundle(authority_set(owners))
    restarted = bundle(authority_set(owners), "case-2")
    assert first.ref_digests() == restarted.ref_digests()
    with pytest.raises(RuntimeError, match="expired"):
        first.validate_active(NOW + timedelta(minutes=6))
