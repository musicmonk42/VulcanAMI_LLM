"""Process-local capabilities for the supported constitutional TCB.

This is not a sandbox against arbitrary code in this interpreter.  Untrusted
models, tools, and plugins must execute out of process.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from enum import Enum
from types import SimpleNamespace


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


class TransitionEdge(str, Enum):
    ADMISSION = "admission"
    VALIDATION = "validation"
    EPISTEMIC_COMMIT = "epistemic_commit"
    PUBLICATION = "publication"


class LiveTransitionPermit:
    __slots__ = (
        "_actor_digest",
        "_edge",
        "_constitution_digest",
        "_episode_id",
        "_expected_prior_episode_digest",
        "_expires_at",
        "_expires_wall",
        "_issuer",
        "_issued_at",
        "_nonce",
        "_policy_digest",
        "_release_digest",
        "_sealed",
        "_snapshot_digest",
        "_validation_digest",
        "_verifier_digest",
    )

    def __init__(self, issuer: object, **facts: object) -> None:
        object.__setattr__(self, "_issuer", issuer)
        object.__setattr__(self, "_nonce", secrets.token_hex(32))
        lifetime = float(facts.pop("lifetime_seconds"))
        object.__setattr__(self, "_issued_at", time.time())
        object.__setattr__(self, "_expires_at", time.monotonic() + lifetime)
        object.__setattr__(self, "_expires_wall", self._issued_at + lifetime)
        for name, value in facts.items():
            object.__setattr__(self, f"_{name}", value)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        raise AttributeError("live transition permits are immutable")

    def __reduce__(self):
        raise TypeError("live transition permits cannot be serialized")

    def __reduce_ex__(self, _protocol):
        return self.__reduce__()

    def __copy__(self):
        raise TypeError("live transition permits cannot be copied")

    def __deepcopy__(self, _memo):
        return self.__copy__()

    @property
    def principal(self):
        """Compatibility projection; descriptive identity grants nothing."""
        return SimpleNamespace(
            identity_digest=self._verifier_digest,
            release_digest=self._release_digest,
            is_kernel=False,
        )

    @property
    def policy_digest(self):
        return self._policy_digest

    @property
    def validation_digest(self):
        return self._validation_digest

    @property
    def snapshot_digest(self):
        return self._snapshot_digest

    @property
    def expected_prior_episode_digest(self):
        return self._expected_prior_episode_digest

    @property
    def grant(self):
        return SimpleNamespace(
            principal_digest=self._verifier_digest,
            evidence_digest=hashlib.sha256(
                (
                    self._validation_digest
                    + self._policy_digest
                    + self._snapshot_digest
                ).encode()
            ).hexdigest(),
        )


class MutationPort:
    """Private issuer/consumer; object identity is the process capability."""

    __slots__ = ("_instance", "_spent", "constitution_digest", "release_digest")

    def __init__(
        self, release_digest: str, constitution_digest: str | None = None
    ) -> None:
        self._instance = object()
        self._spent: set[str] = set()
        self.release_digest = _digest(release_digest, "qualified release digest")
        self.constitution_digest = (
            _digest(constitution_digest, "constitution digest")
            if constitution_digest is not None
            else hashlib.sha256(b"vulcan-constitution-v1").hexdigest()
        )

    def issue(self, *, edge: TransitionEdge, lifetime_seconds: float = 30, **facts):
        if not isinstance(edge, TransitionEdge):
            raise TypeError("a typed transition edge is required")
        if lifetime_seconds <= 0 or lifetime_seconds > 300:
            raise ValueError("permit lifetime is outside the supported bound")
        required = {
            "actor_digest",
            "episode_id",
            "expected_prior_episode_digest",
            "policy_digest",
            "snapshot_digest",
            "validation_digest",
            "verifier_digest",
        }
        if (
            set(facts) != required
            or not isinstance(facts["episode_id"], str)
            or not 1 <= len(facts["episode_id"]) <= 128
        ):
            raise ValueError("permit facts are incomplete")
        for name in required - {"episode_id"}:
            _digest(facts[name], name.replace("_", " "))
        return LiveTransitionPermit(
            self._instance,
            edge=edge,
            constitution_digest=self.constitution_digest,
            release_digest=self.release_digest,
            lifetime_seconds=lifetime_seconds,
            **facts,
        )

    def consume(self, permit: LiveTransitionPermit, *, edge: TransitionEdge, **facts):
        if type(permit) is not LiveTransitionPermit:
            raise PermissionError("a live transition permit is required")
        if permit._issuer is not self._instance or permit._edge is not edge:
            raise PermissionError("permit issuer or edge mismatch")
        if permit._nonce in self._spent or time.monotonic() > permit._expires_at:
            raise PermissionError("permit is consumed or expired")
        for name, expected in facts.items():
            if getattr(permit, f"_{name}", None) != expected:
                raise PermissionError(f"permit {name} binding mismatch")
        self._spent.add(permit._nonce)
        return {
            "edge": permit._edge.value,
            "constitution_digest": permit._constitution_digest,
            "expires_at_epoch": permit._expires_wall,
            "issued_at_epoch": permit._issued_at,
            "nonce_digest": hashlib.sha256(permit._nonce.encode()).hexdigest(),
            "policy_digest": permit._policy_digest,
            "qualified_release_digest": permit._release_digest,
            "snapshot_digest": permit._snapshot_digest,
            "validation_digest": permit._validation_digest,
            "verifier_digest": permit._verifier_digest,
        }
